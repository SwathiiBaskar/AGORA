"""The error envelope returned by every non-2xx API response.

Every failure, whether raised by a route, rejected by validation, or unexpected,
is rendered as the same JSON shape::

    {"error": {"code": "not_found", "message": "Not Found", "request_id": "req_..."}}

``code`` is a stable machine-readable category, ``message`` is safe to show to an
end user, and ``request_id`` matches the ``X-Request-ID`` response header and the
server log lines for the request. Internal details (exception text, stack traces,
rejected input values) never appear in a response.
"""

from http import HTTPStatus
from typing import Final

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from civbox.core.errors import CivboxError
from civbox.core.ids import new_request_id
from civbox.core.redaction import redact_string

MAX_VALIDATION_DETAILS: Final = 50
INTERNAL_ERROR_MESSAGE: Final = "An internal error occurred."

_CODE_BY_STATUS: Final = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    415: "unsupported_media_type",
    422: "validation_error",
    429: "rate_limited",
}


class ErrorDetail(BaseModel):
    """One problem found while validating a request."""

    location: list[str | int]
    message: str
    type: str


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str
    details: list[ErrorDetail] | None = None


class ErrorResponse(BaseModel):
    """Envelope for every error response."""

    error: ErrorBody


def code_for_status(status_code: int) -> str:
    return _CODE_BY_STATUS.get(status_code, f"http_{status_code}")


def error_response(
    status_code: int,
    code: str,
    message: str,
    request_id: str,
    *,
    details: list[ErrorDetail] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorBody(code=code, message=message, request_id=request_id, details=details)
    )
    return JSONResponse(
        body.model_dump(mode="json", exclude_none=True),
        status_code=status_code,
        headers=headers,
    )


def internal_error_response(request_id: str, error: BaseException) -> JSONResponse:
    """Response for an unexpected failure. Reveals only the error's category code."""
    code = error.code if isinstance(error, CivboxError) else "internal_error"
    return error_response(500, code, INTERNAL_ERROR_MESSAGE, request_id)


def request_id_of(request: Request) -> str:
    state = request.scope.get("state", {})
    value = state.get("request_id") if isinstance(state, dict) else None
    return value if isinstance(value, str) else new_request_id()


def _status_phrase(status_code: int) -> str:
    try:
        return HTTPStatus(status_code).phrase
    except ValueError:
        return "Error"


def register_exception_handlers(app: FastAPI) -> None:
    """Render framework-raised errors in the standard envelope."""

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        message = exc.detail if isinstance(exc.detail, str) else _status_phrase(exc.status_code)
        return error_response(
            exc.status_code,
            code_for_status(exc.status_code),
            redact_string(message),
            request_id_of(request),
            headers=dict(exc.headers) if exc.headers else None,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Only the location, message and type of each problem are reported. The
        # rejected input is never echoed back, because it may be a credential.
        details = [
            ErrorDetail(
                location=[
                    part if isinstance(part, str | int) else str(part) for part in error["loc"]
                ],
                message=redact_string(str(error["msg"])),
                type=str(error["type"]),
            )
            for error in exc.errors()[:MAX_VALIDATION_DETAILS]
        ]
        return error_response(
            422,
            code_for_status(422),
            "Request validation failed",
            request_id_of(request),
            details=details,
        )
