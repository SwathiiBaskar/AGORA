"""Structured JSON logging.

Every record becomes one JSON object on one line, carrying the service and
environment plus whichever correlation identifiers are bound in the current
context (see :func:`log_context`). Message text, extra fields and tracebacks all
pass through :mod:`civbox.core.redaction` before they are written.

Logs go to **stderr** by default so that command output on stdout (for example
``civbox config show``) stays machine-readable when piped.
"""

import json
import logging
import sys
import traceback
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Final, TextIO

from civbox.core.redaction import redact, redact_string

# The closed set of correlation identifiers. A closed set keeps the log schema
# stable and prevents accidental high-cardinality or sensitive context fields.
CORRELATION_FIELDS: Final = (
    "request_id",
    "trace_id",
    "job_id",
    "simulation_id",
    "experiment_id",
    "event_id",
)

_MAX_TRACEBACK: Final = 8000
_HANDLER_MARKER: Final = "_civbox_handler"

_context: ContextVar[Mapping[str, str]] = ContextVar(
    "civbox_log_context", default=MappingProxyType({})
)

# Attributes every LogRecord has; anything else on a record came from ``extra=``.
_RESERVED_ATTRIBUTES: Final = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


@contextmanager
def log_context(
    *,
    request_id: str | None = None,
    trace_id: str | None = None,
    job_id: str | None = None,
    simulation_id: str | None = None,
    experiment_id: str | None = None,
    event_id: str | None = None,
) -> Iterator[None]:
    """Bind correlation identifiers to every log record emitted inside the block.

    Contexts nest (inner values override outer ones for the same field) and are
    isolated per thread and per asyncio task, because they use ``contextvars``.
    Fields passed as ``None`` are left unchanged.
    """
    provided = {
        "request_id": request_id,
        "trace_id": trace_id,
        "job_id": job_id,
        "simulation_id": simulation_id,
        "experiment_id": experiment_id,
        "event_id": event_id,
    }
    merged = {**_context.get(), **{k: v for k, v in provided.items() if v is not None}}
    token = _context.set(MappingProxyType(merged))
    try:
        yield
    finally:
        _context.reset(token)


def current_context() -> Mapping[str, str]:
    """Return the correlation identifiers bound in the current context."""
    return _context.get()


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON with redaction applied."""

    def __init__(self, *, service: str, environment: str) -> None:
        super().__init__()
        self._service = service
        self._environment = environment

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": _iso_timestamp(record.created),
            "level": record.levelname,
            "logger": record.name,
            "message": redact_string(record.getMessage()),
            "service": self._service,
            "environment": self._environment,
        }
        payload.update(current_context())

        extras = {k: v for k, v in record.__dict__.items() if k not in _RESERVED_ATTRIBUTES}
        redacted_extras = redact(extras)
        if isinstance(redacted_extras, dict):
            for key, value in redacted_extras.items():
                # A caller-supplied field must never overwrite a base field.
                payload[f"extra_{key}" if key in payload else key] = value

        if record.exc_info is not None and record.exc_info[1] is not None:
            exc_type, exc_value, exc_tb = record.exc_info
            payload["exception"] = {
                "type": exc_type.__name__ if exc_type is not None else "Exception",
                "message": redact_string(str(exc_value)),
                "traceback": redact_string(
                    "".join(traceback.format_exception(exc_type, exc_value, exc_tb)),
                    _MAX_TRACEBACK,
                ),
            }

        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)


def configure_logging(
    *,
    service: str,
    environment: str,
    level: str = "INFO",
    stream: TextIO | None = None,
) -> None:
    """Install the JSON handler on the root logger.

    Safe to call more than once: a previously installed civbox handler is replaced,
    never duplicated. Raises ``ValueError`` for an unknown level name.
    """
    if level.upper() not in logging.getLevelNamesMapping():
        raise ValueError(f"unknown log level: {level!r}")

    root = logging.getLogger()
    for existing in list(root.handlers):
        if getattr(existing, _HANDLER_MARKER, False):
            root.removeHandler(existing)

    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(JsonFormatter(service=service, environment=environment))
    setattr(handler, _HANDLER_MARKER, True)
    root.addHandler(handler)
    root.setLevel(level.upper())


def _iso_timestamp(created: float) -> str:
    moment = datetime.fromtimestamp(created, tz=UTC)
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")
