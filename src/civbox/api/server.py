"""Process entry point for serving the public API with uvicorn."""

import logging

import uvicorn
from starlette.types import ASGIApp

from civbox.api.app import create_app
from civbox.core.config import Settings
from civbox.core.logging import configure_logging


class _DropColorMessage(logging.Filter):
    """Remove uvicorn's ``color_message`` field.

    uvicorn attaches a second copy of some messages containing ANSI colour codes. It
    is useful on a terminal and noise in structured logs.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.__dict__.pop("color_message", None)
        return True


_DROP_COLOR_MESSAGE = _DropColorMessage()


def build_server_config(app: ASGIApp, *, host: str, port: int, log_level: str) -> uvicorn.Config:
    """Return the uvicorn configuration used to serve ``app``.

    uvicorn's own logging setup and access log are disabled: records flow through the
    application's JSON handler instead, and the request-context middleware writes the
    access line, so each request is logged exactly once, with its request ID. The
    ``Server`` header is suppressed to avoid advertising the server software.
    """
    return uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level=log_level.lower(),
        log_config=None,
        access_log=False,
        server_header=False,
    )


def serve(settings: Settings, *, host: str, port: int) -> None:
    """Configure logging and serve the public API until interrupted."""
    configure_logging(service="civbox-api", environment=settings.app_env, level=settings.log_level)
    # addFilter ignores a filter that is already attached, so repeated calls are safe.
    logging.getLogger("uvicorn.error").addFilter(_DROP_COLOR_MESSAGE)
    app = create_app(settings)
    uvicorn.Server(
        build_server_config(app, host=host, port=port, log_level=settings.log_level)
    ).run()
