"""Liveness and readiness endpoints.

* ``GET /healthz`` answers whether the process is running and able to serve a
  request. It checks no dependencies, so a dependency outage never causes the
  process to be restarted.
* ``GET /readyz`` answers whether the process can do useful work right now. It runs
  every registered readiness check concurrently, each bounded by a timeout, and
  returns 503 if any fails. The application currently has no external dependencies,
  so no checks are registered and it is always ready.

Failure reasons are logged, never returned: the response names which checks
failed but not why.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import Final, Literal, cast

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("civbox.api.health")

ReadinessCheck = Callable[[], Awaitable[bool]]

READINESS_CHECK_TIMEOUT_SECONDS: Final = 2.0

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, Literal["ok", "failed"]]


@router.get("/healthz", response_model=HealthResponse, summary="Liveness")
async def healthz() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get(
    "/readyz",
    response_model=ReadinessResponse,
    responses={503: {"model": ReadinessResponse, "description": "A dependency is unavailable"}},
    summary="Readiness",
)
async def readyz(request: Request) -> Response:
    checks = cast(Mapping[str, ReadinessCheck], request.app.state.readiness_checks)
    outcomes = await asyncio.gather(*(_run_check(name, check) for name, check in checks.items()))
    report = ReadinessResponse(
        status="ready" if all(ok for _, ok in outcomes) else "not_ready",
        checks={name: "ok" if ok else "failed" for name, ok in outcomes},
    )
    return JSONResponse(report.model_dump(), status_code=200 if report.status == "ready" else 503)


async def _run_check(name: str, check: ReadinessCheck) -> tuple[str, bool]:
    try:
        ok = await asyncio.wait_for(check(), timeout=READINESS_CHECK_TIMEOUT_SECONDS)
    except Exception:
        logger.warning("readiness check failed", extra={"check": name}, exc_info=True)
        return name, False
    return name, bool(ok)
