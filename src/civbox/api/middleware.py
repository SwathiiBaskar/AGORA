"""ASGI middleware for the public API.

Both classes are plain ASGI middleware rather than ``BaseHTTPMiddleware``: the
latter runs the downstream application in a separate task and wraps its response
stream, which complicates exception handling and streaming responses.

Stack order matters. :class:`SecurityHeadersMiddleware` is outermost so that every
response, including the 500 produced by :class:`RequestContextMiddleware` for an
unhandled exception, carries the security headers. The framework's own
last-resort error handler sits outside both and would bypass them, which is why
unhandled exceptions are caught here and rendered by this layer instead.
"""

import logging
import time
from typing import Final

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from civbox.api.errors import internal_error_response
from civbox.core.ids import is_valid_request_id, new_request_id
from civbox.core.logging import log_context

REQUEST_ID_HEADER: Final = "X-Request-ID"

# Probe endpoints are polled constantly; their access lines are kept at DEBUG.
_QUIET_PATHS: Final = frozenset({"/healthz", "/readyz"})

SECURITY_HEADERS: Final = (
    ("X-Content-Type-Options", "nosniff"),
    ("X-Frame-Options", "DENY"),
    ("Referrer-Policy", "no-referrer"),
    # The API only ever returns JSON, so nothing may load or embed.
    ("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'"),
    ("Cache-Control", "no-store"),
)
HSTS_HEADER: Final = ("Strict-Transport-Security", "max-age=63072000; includeSubDomains")

_access_log = logging.getLogger("civbox.access")
_error_log = logging.getLogger("civbox.api")


class RequestContextMiddleware:
    """Identify each request, log it, and answer unhandled exceptions with a 500.

    A caller-supplied ``X-Request-ID`` is adopted only if it passes
    :func:`civbox.core.ids.is_valid_request_id`; otherwise a fresh one is generated.
    The identifier is stored in ``scope["state"]``, bound to every log record emitted
    while the request is handled, and returned in the response header.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        supplied = Headers(scope=scope).get(REQUEST_ID_HEADER)
        request_id = supplied if supplied and is_valid_request_id(supplied) else new_request_id()
        scope.setdefault("state", {})["request_id"] = request_id

        started_at = time.perf_counter()
        status_code = 500
        response_started = False

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code, response_started
            if message["type"] == "http.response.start":
                response_started = True
                status_code = message["status"]
                message.setdefault("headers", [])
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        method = scope["method"]
        path = scope["path"]
        with log_context(request_id=request_id):
            try:
                await self.app(scope, receive, send_with_request_id)
            except Exception as exc:
                _error_log.exception(
                    "unhandled exception while handling request",
                    extra={"http_method": method, "http_path": path},
                )
                if response_started:
                    # Headers are already on the wire; the connection cannot be repaired.
                    raise
                await internal_error_response(request_id, exc)(scope, receive, send_with_request_id)
            finally:
                _access_log.log(
                    logging.DEBUG if path in _QUIET_PATHS else logging.INFO,
                    "request handled",
                    extra={
                        "http_method": method,
                        "http_path": path,
                        "http_status": status_code,
                        "duration_ms": round((time.perf_counter() - started_at) * 1000, 3),
                    },
                )


class SecurityHeadersMiddleware:
    """Add defensive response headers without overriding anything a route set."""

    def __init__(self, app: ASGIApp, *, hsts: bool) -> None:
        self.app = app
        self._headers = (*SECURITY_HEADERS, HSTS_HEADER) if hsts else SECURITY_HEADERS

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_security_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                message.setdefault("headers", [])
                headers = MutableHeaders(scope=message)
                for name, value in self._headers:
                    headers.setdefault(name, value)
            await send(message)

        await self.app(scope, receive, send_with_security_headers)
