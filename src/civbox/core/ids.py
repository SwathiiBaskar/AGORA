"""Identifier generation and validation."""

import re
import uuid
from typing import Final

# Accepting caller-supplied request identifiers is useful for tracing across
# systems, but they end up in logs and response headers. The character set is
# therefore restricted to values that cannot carry control characters, spaces,
# or markup, and the length is bounded.
_REQUEST_ID: Final = re.compile(r"[A-Za-z0-9._-]{8,128}")


def new_request_id() -> str:
    """Return a new, unguessable request identifier such as ``req_<32 hex digits>``."""
    return f"req_{uuid.uuid4().hex}"


def is_valid_request_id(value: str) -> bool:
    """Return True if ``value`` is safe to adopt as a request identifier."""
    return _REQUEST_ID.fullmatch(value) is not None
