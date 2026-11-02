"""Behaviour of the API as served by uvicorn on a real socket.

In-process tests cannot see what the server software does to the traffic, such as
adding a ``Server`` header or writing its own access log. These tests assert on the
bytes a real HTTP client receives.
"""

import threading
import time
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi import FastAPI

from civbox.api.app import create_app
from civbox.api.middleware import SECURITY_HEADERS
from civbox.api.server import build_server_config
from civbox.core.config import load_settings

from .support import LogEntries

STARTUP_TIMEOUT_SECONDS = 15.0


@pytest.fixture
def base_url(json_logs: LogEntries) -> Iterator[str]:
    app: FastAPI = create_app(load_settings(env_file=None))

    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("detail postgresql://app:hunter2@db/x")

    server = uvicorn.Server(build_server_config(app, host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, name="test-api-server", daemon=True)
    thread.start()

    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while not server.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            raise RuntimeError("test server did not start")
        time.sleep(0.01)

    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=15)
        assert not thread.is_alive(), "test server did not shut down"


def get(url: str, **kwargs: Any) -> httpx.Response:
    return httpx.get(url, timeout=10, **kwargs)


def wait_for(predicate: Any, what: str) -> None:
    deadline = time.monotonic() + 10
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.01)


def test_a_caller_supplied_request_id_round_trips_over_the_wire(base_url: str) -> None:
    response = get(f"{base_url}/healthz", headers={"X-Request-ID": "wire-trace-0001"})
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["x-request-id"] == "wire-trace-0001"


def test_the_wire_response_has_security_headers_and_hides_the_server_software(
    base_url: str,
) -> None:
    response = get(f"{base_url}/healthz")
    for name, value in SECURITY_HEADERS:
        assert response.headers[name] == value
    assert "server" not in response.headers


def test_an_unhandled_error_returns_the_envelope_and_the_server_keeps_serving(
    base_url: str,
) -> None:
    failed = get(f"{base_url}/boom")
    assert failed.status_code == 500
    assert failed.json()["error"]["code"] == "internal_error"
    assert failed.json()["error"]["request_id"] == failed.headers["x-request-id"]
    assert "hunter2" not in failed.text
    assert get(f"{base_url}/healthz").status_code == 200


def test_each_request_is_logged_exactly_once_with_its_request_id(
    base_url: str, json_logs: LogEntries
) -> None:
    response = get(f"{base_url}/nope", headers={"X-Request-ID": "wire-trace-0002"})
    assert response.status_code == 404
    # A subsequent request is logged strictly after this one, so once it is visible any
    # duplicate line for the first request would be visible too.
    get(f"{base_url}/nope", headers={"X-Request-ID": "wire-trace-0003"})
    wait_for(
        lambda: any(e.get("request_id") == "wire-trace-0003" for e in json_logs()),
        "the access line of the second request",
    )

    lines = [e for e in json_logs() if e.get("request_id") == "wire-trace-0002"]
    assert [e["logger"] for e in lines] == ["civbox.access"]
    assert lines[0]["http_status"] == 404
    assert not any(e["logger"].startswith("uvicorn") for e in json_logs() if e["level"] == "INFO")


def test_the_server_logs_the_unhandled_error_without_leaking_secrets(
    base_url: str, json_logs: LogEntries
) -> None:
    get(f"{base_url}/boom", headers={"X-Request-ID": "wire-trace-0004"})
    wait_for(
        lambda: any(
            e["logger"] == "civbox.access" and e.get("request_id") == "wire-trace-0004"
            for e in json_logs()
        ),
        "the access line for the failed request",
    )
    errors = [e for e in json_logs() if e["level"] == "ERROR"]
    assert [e["request_id"] for e in errors] == ["wire-trace-0004"]
    assert "hunter2" not in str(json_logs())
