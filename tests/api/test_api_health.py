import asyncio

import pytest

from civbox.api import health
from civbox.api.app import create_app
from civbox.core.config import Settings

from .support import AppClient, LogEntries


def client_with(**checks: health.ReadinessCheck) -> AppClient:
    return AppClient(create_app(Settings(_env_file=None), readiness_checks=checks))


async def healthy() -> bool:
    return True


async def unhealthy() -> bool:
    return False


async def raises() -> bool:
    raise ConnectionError("secret detail postgresql://app:hunter2@db/x")


def test_liveness_reports_ok_and_checks_no_dependencies() -> None:
    client = client_with(database=raises)  # a broken dependency must not affect liveness
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_without_registered_checks_is_ready(client: AppClient) -> None:
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "checks": {}}


def test_readiness_reports_each_passing_check() -> None:
    response = client_with(database=healthy, cache=healthy).get("/readyz")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "checks": {"database": "ok", "cache": "ok"}}


def test_a_failing_check_makes_the_service_not_ready() -> None:
    response = client_with(database=healthy, cache=unhealthy).get("/readyz")
    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "checks": {"database": "ok", "cache": "failed"},
    }


def test_a_raising_check_fails_without_leaking_the_reason(json_logs: LogEntries) -> None:
    response = client_with(database=raises).get("/readyz")
    assert response.status_code == 503
    assert response.json()["checks"] == {"database": "failed"}
    assert "hunter2" not in response.text
    assert "ConnectionError" not in response.text
    warnings = [e for e in json_logs() if e["logger"] == "civbox.api.health"]
    assert [(w["level"], w["check"]) for w in warnings] == [("WARNING", "database")]
    assert "hunter2" not in str(json_logs())


def test_a_check_that_exceeds_the_timeout_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(health, "READINESS_CHECK_TIMEOUT_SECONDS", 0.05)

    async def hangs() -> bool:
        await asyncio.sleep(30)
        return True

    response = client_with(database=hangs).get("/readyz")
    assert response.status_code == 503
    assert response.json()["checks"] == {"database": "failed"}


def test_checks_run_concurrently(monkeypatch: pytest.MonkeyPatch) -> None:
    # Each check can only finish after the other has started. Run one after the
    # other, the first would wait out its timeout and fail.
    monkeypatch.setattr(health, "READINESS_CHECK_TIMEOUT_SECONDS", 1.0)
    first_started, second_started = asyncio.Event(), asyncio.Event()

    async def first() -> bool:
        first_started.set()
        await second_started.wait()
        return True

    async def second() -> bool:
        second_started.set()
        await first_started.wait()
        return True

    response = client_with(first=first, second=second).get("/readyz")
    assert response.status_code == 200
    assert response.json()["checks"] == {"first": "ok", "second": "ok"}


def test_readiness_failure_is_documented_in_openapi(client: AppClient) -> None:
    responses = client.get("/openapi.json").json()["paths"]["/readyz"]["get"]["responses"]
    assert responses["503"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/ReadinessResponse"
    }


def test_registered_checks_are_isolated_per_application() -> None:
    first = create_app(Settings(_env_file=None), readiness_checks={"a": unhealthy})
    second = create_app(Settings(_env_file=None))
    assert AppClient(first).get("/readyz").status_code == 503
    assert AppClient(second).get("/readyz").status_code == 200
