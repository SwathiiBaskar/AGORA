import asyncio
import logging
import re
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Response
from starlette.types import Message, Receive, Scope, Send

from civbox.api.app import create_app
from civbox.api.middleware import (
    HSTS_HEADER,
    SECURITY_HEADERS,
    RequestContextMiddleware,
    SecurityHeadersMiddleware,
)
from civbox.core.config import Settings

from .support import AppClient, LogEntries

GENERATED_ID = re.compile(r"req_[0-9a-f]{32}")


def access_entries(logs: LogEntries) -> list[dict[str, Any]]:
    return [entry for entry in logs() if entry["logger"] == "civbox.access"]


# --- request identification -------------------------------------------------


def test_each_response_carries_a_freshly_generated_request_id(client: AppClient) -> None:
    first = client.get("/healthz").headers["x-request-id"]
    second = client.get("/healthz").headers["x-request-id"]
    assert GENERATED_ID.fullmatch(first)
    assert GENERATED_ID.fullmatch(second)
    assert first != second


def test_a_valid_caller_supplied_request_id_is_adopted(client: AppClient) -> None:
    response = client.get("/nope", headers={"X-Request-ID": "trace-12345.abc"})
    assert response.headers["x-request-id"] == "trace-12345.abc"
    assert response.json()["error"]["request_id"] == "trace-12345.abc"


@pytest.mark.parametrize(
    "supplied", ["short", "has space here", "x" * 129, "bad/char!", "<b>id</b>"]
)
def test_an_unsafe_caller_supplied_request_id_is_replaced(client: AppClient, supplied: str) -> None:
    response = client.get("/healthz", headers={"X-Request-ID": supplied})
    assert GENERATED_ID.fullmatch(response.headers["x-request-id"])


def test_log_records_written_by_a_route_carry_the_request_id(
    app: FastAPI, client: AppClient, json_logs: LogEntries
) -> None:
    @app.get("/work")
    async def work() -> dict[str, bool]:
        logging.getLogger("civbox.test").info("inside the handler")
        return {"done": True}

    response = client.get("/work")
    inside = [e for e in json_logs() if e["message"] == "inside the handler"]
    assert [e["request_id"] for e in inside] == [response.headers["x-request-id"]]


def test_concurrent_requests_never_share_a_request_id(app: FastAPI, json_logs: LogEntries) -> None:
    @app.get("/slow/{n}")
    async def slow(n: int) -> dict[str, int]:
        await asyncio.sleep(0.001 * (20 - n))  # finish in reverse order of arrival
        logging.getLogger("civbox.test").info("handled %d", n)
        return {"n": n}

    async def call_all() -> list[httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            return list(await asyncio.gather(*(c.get(f"/slow/{n}") for n in range(20))))

    responses = asyncio.run(call_all())
    logged = {e["message"]: e["request_id"] for e in json_logs() if e["logger"] == "civbox.test"}
    for n, response in enumerate(responses):
        assert response.json() == {"n": n}
        assert logged[f"handled {n}"] == response.headers["x-request-id"]
    assert len({r.headers["x-request-id"] for r in responses}) == 20


# --- access log -------------------------------------------------------------


def test_every_request_writes_one_access_line(client: AppClient, json_logs: LogEntries) -> None:
    response = client.get("/nope")
    (entry,) = access_entries(json_logs)
    assert entry["request_id"] == response.headers["x-request-id"]
    assert entry["http_method"] == "GET"
    assert entry["http_path"] == "/nope"
    assert entry["http_status"] == 404
    assert entry["duration_ms"] >= 0
    assert entry["level"] == "INFO"


def test_the_query_string_is_never_logged(client: AppClient, json_logs: LogEntries) -> None:
    client.get("/healthz", params={"token": "sup3r-s3cret-value"})
    assert "sup3r-s3cret-value" not in str(json_logs())
    (entry,) = access_entries(json_logs)
    assert entry["http_path"] == "/healthz"


def test_probe_endpoints_are_logged_at_debug(client: AppClient, json_logs: LogEntries) -> None:
    client.get("/healthz")
    client.get("/readyz")
    assert [e["level"] for e in access_entries(json_logs)] == ["DEBUG", "DEBUG"]


def test_unexpected_failures_are_logged_as_status_500(
    app: FastAPI, client: AppClient, json_logs: LogEntries
) -> None:
    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("x")

    client.get("/boom")
    (entry,) = access_entries(json_logs)
    assert entry["http_status"] == 500


# --- exception handling at the ASGI level ----------------------------------


async def run_asgi(app: Any, scope: Scope) -> list[Message]:
    sent: list[Message] = []

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        sent.append(message)

    await app(scope, receive, send)
    return sent


def http_scope() -> Scope:
    return {"type": "http", "method": "GET", "path": "/stream", "headers": []}


def test_an_exception_after_the_response_started_is_logged_and_propagated(
    json_logs: LogEntries,
) -> None:
    async def failing_app(scope: Scope, receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        raise RuntimeError("stream broke")

    sent: list[Message] = []

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        sent.append(message)

    middleware = RequestContextMiddleware(failing_app)
    with pytest.raises(RuntimeError, match="stream broke"):
        asyncio.run(middleware(http_scope(), receive, send))

    assert len(sent) == 1  # no second response was attempted
    assert any(name == b"x-request-id" for name, _ in sent[0]["headers"])
    assert [e["level"] for e in json_logs() if e["logger"] == "civbox.api"] == ["ERROR"]


def test_non_http_scopes_pass_through_untouched() -> None:
    seen: list[Scope] = []

    async def inner(scope: Scope, receive: Receive, send: Send) -> None:
        seen.append(scope)

    for middleware in (
        RequestContextMiddleware(inner),
        SecurityHeadersMiddleware(inner, hsts=True),
    ):
        scope: Scope = {"type": "lifespan"}
        asyncio.run(run_asgi(middleware, scope))
        assert seen[-1] is scope
        assert "state" not in scope


# --- security headers -------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path"),
    [("GET", "/healthz"), ("GET", "/nope"), ("POST", "/healthz"), ("GET", "/items")],
)
def test_security_headers_are_present_on_every_kind_of_response(
    app: FastAPI, client: AppClient, method: str, path: str
) -> None:
    @app.get("/items")
    async def items(limit: int) -> dict[str, int]:
        return {"limit": limit}

    response = client.request(method, path)
    for name, value in SECURITY_HEADERS:
        assert response.headers[name] == value


def test_security_headers_are_present_on_unhandled_errors(app: FastAPI, client: AppClient) -> None:
    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("x")

    response = client.get("/boom")
    assert response.status_code == 500
    for name, value in SECURITY_HEADERS:
        assert response.headers[name] == value
    assert "x-request-id" in response.headers


def test_responses_forbid_framing_and_loading_anything(client: AppClient) -> None:
    headers = client.get("/healthz").headers
    assert "default-src 'none'" in headers["content-security-policy"]
    assert "frame-ancestors 'none'" in headers["content-security-policy"]
    assert headers["x-content-type-options"] == "nosniff"


def test_transport_security_is_only_sent_in_production() -> None:
    development = AppClient(create_app(Settings(app_env="development", _env_file=None)))
    production = AppClient(create_app(Settings(app_env="production", _env_file=None)))
    name, value = HSTS_HEADER
    assert name.lower() not in development.get("/healthz").headers
    assert production.get("/healthz").headers[name] == value


def test_a_route_may_override_a_default_header(app: FastAPI, client: AppClient) -> None:
    @app.get("/cached")
    async def cached(response: Response) -> dict[str, bool]:
        response.headers["Cache-Control"] = "max-age=60"
        return {"ok": True}

    assert client.get("/cached").headers["cache-control"] == "max-age=60"
    assert client.get("/healthz").headers["cache-control"] == "no-store"
