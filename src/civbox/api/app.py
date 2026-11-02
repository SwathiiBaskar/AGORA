"""Application factory for the public API."""

from collections.abc import Mapping
from typing import Any

from fastapi import FastAPI

from civbox import __version__
from civbox.api.errors import ErrorResponse, register_exception_handlers
from civbox.api.health import ReadinessCheck
from civbox.api.health import router as health_router
from civbox.api.middleware import RequestContextMiddleware, SecurityHeadersMiddleware
from civbox.core.config import Settings

_ERROR_SCHEMA_REF = "#/components/schemas/ErrorResponse"


def create_app(
    settings: Settings,
    *,
    readiness_checks: Mapping[str, ReadinessCheck] | None = None,
) -> FastAPI:
    """Build the public API application.

    ``readiness_checks`` maps a dependency name to an async callable returning True
    when that dependency is usable; ``/readyz`` reports 503 while any returns False,
    raises, or exceeds its timeout.

    The interactive documentation pages are disabled: they load scripts from a CDN
    and would require weakening the Content-Security-Policy. The OpenAPI document at
    ``/openapi.json`` is the API contract.
    """
    app = FastAPI(
        title="Civilization in a Box API",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url="/openapi.json",
        responses={500: {"model": ErrorResponse, "description": "Unexpected server error"}},
    )
    app.state.readiness_checks = dict(readiness_checks or {})

    register_exception_handlers(app)
    app.include_router(health_router)

    # Added last means outermost: security headers must also cover the 500 that the
    # request-context layer renders for unhandled exceptions.
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(SecurityHeadersMiddleware, hsts=settings.app_env == "production")

    _document_validation_errors_as_envelope(app)
    return app


def normalise_openapi(schema: dict[str, Any]) -> dict[str, Any]:
    """Make the OpenAPI document describe the validation errors the API really returns.

    FastAPI documents 422 responses with its own ``HTTPValidationError`` schema, but
    this API answers with the standard error envelope. Left alone, generated clients
    would be typed against a shape the server never sends.
    """
    for path_item in schema.get("paths", {}).values():
        for operation in path_item.values():
            if not isinstance(operation, dict):
                continue
            response = operation.get("responses", {}).get("422")
            if response is not None:
                response["description"] = "Request validation failed"
                response["content"] = {"application/json": {"schema": {"$ref": _ERROR_SCHEMA_REF}}}
    components = schema.get("components", {}).get("schemas", {})
    for unused in ("HTTPValidationError", "ValidationError"):
        components.pop(unused, None)
    return schema


def _document_validation_errors_as_envelope(app: FastAPI) -> None:
    generate = app.openapi

    def openapi() -> dict[str, Any]:
        return normalise_openapi(generate())

    app.openapi = openapi  # type: ignore[method-assign]
