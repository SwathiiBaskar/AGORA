"""Redaction of secrets from anything that is about to be logged or displayed.

Two complementary mechanisms, because neither is sufficient alone:

* **Key based.** A mapping entry whose key names a credential (``password``,
  ``GEMINI_API_KEY``, ``accessToken`` ...) has its whole value replaced.
* **Pattern based.** Free text is scanned for well-known secret shapes: credentials
  embedded in URLs, bearer tokens, JWTs, Google API keys, and ``name=value`` pairs
  whose name looks like a credential.

This is defence in depth, not a guarantee. A secret with no recognisable key or
shape that is interpolated into a message will not be caught. Keep secrets in
``SecretStr`` fields and never format them into text.

:func:`redact` also returns a JSON-safe, size-bounded copy of its input, so the same
call protects log pipelines from huge payloads (prompt bodies, documents).
"""

import itertools
import math
import re
from collections.abc import Mapping
from enum import Enum
from typing import Final

REDACTED: Final = "[REDACTED]"

DEFAULT_MAX_STRING: Final = 2000
DEFAULT_MAX_DEPTH: Final = 8
DEFAULT_MAX_ITEMS: Final = 200

# Text longer than this is cut before pattern matching so scan cost stays bounded.
_SCAN_LIMIT: Final = 20_000

# Splits "accessToken", "GEMINI_API_KEY" and "x-api-key" into lower-case segments.
_SEGMENT_SPLIT: Final = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|[^A-Za-z0-9]+")

_SENSITIVE_SEGMENTS: Final = frozenset(
    {
        "password",
        "passwd",
        "passphrase",
        "secret",
        "secrets",
        "credential",
        "credentials",
        "authorization",
        "cookie",
        "cookies",
        "apikey",
    }
)

_SENSITIVE_PAIRS: Final = frozenset(
    {
        ("api", "key"),
        ("access", "key"),
        ("auth", "key"),
        ("secret", "key"),
        ("private", "key"),
        ("signing", "key"),
        ("hmac", "key"),
        ("encryption", "key"),
    }
)

# Every pattern below is anchored on a literal keyword or scheme, and any run that
# could be repeated from many start positions is length-bounded, so adversarial long
# inputs cannot cause quadratic work.
_URL_CREDENTIALS: Final = re.compile(
    r"(?P<head>\b[A-Za-z][A-Za-z0-9+.\-]{0,31}://[^\s:/@]*:)(?P<secret>[^\s@/]+)(?P<tail>@)"
)
_BEARER: Final = re.compile(r"(?P<head>\bbearer\s+)[A-Za-z0-9\-._~+/]{8,}=*", re.IGNORECASE)
_JWT: Final = re.compile(r"\beyJ[A-Za-z0-9_\-]{5,}\.[A-Za-z0-9_\-]{5,}\.[A-Za-z0-9_\-]*")
_GOOGLE_API_KEY: Final = re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")
_ASSIGNMENT: Final = re.compile(
    r"(?P<head>(?:(?:password|passwd|passphrase|secret|api[_\-]?key|authorization"
    r"|cookie|credentials?)[\w\-.]{0,64}|token\b)[\"']?\s*[=:]\s*)"
    r"(?P<value>\"[^\"]*\"|'[^']*'|[^\s,;&\"']+)",
    re.IGNORECASE,
)


def is_sensitive_key(name: str) -> bool:
    """Return True if ``name`` looks like the name of a credential.

    Singular ``token`` is a credential (``access_token``); plural ``tokens`` is a
    count (``input_tokens``) and is not redacted, because LLM telemetry depends on it.
    """
    segments = [part.lower() for part in _SEGMENT_SPLIT.split(name) if part]
    if not segments:
        return False
    if any(part in _SENSITIVE_SEGMENTS for part in segments):
        return True
    if segments[-1] == "token":
        return True
    return any(pair in _SENSITIVE_PAIRS for pair in itertools.pairwise(segments))


def redact_text(text: str) -> str:
    """Mask well-known secret shapes inside free text. Length is not limited."""
    scanned = text if len(text) <= _SCAN_LIMIT else text[:_SCAN_LIMIT]
    scanned = _URL_CREDENTIALS.sub(rf"\g<head>{REDACTED}\g<tail>", scanned)
    scanned = _BEARER.sub(rf"\g<head>{REDACTED}", scanned)
    scanned = _JWT.sub(REDACTED, scanned)
    scanned = _GOOGLE_API_KEY.sub(REDACTED, scanned)
    scanned = _ASSIGNMENT.sub(rf"\g<head>{REDACTED}", scanned)
    if len(text) > _SCAN_LIMIT:
        return scanned + f"...[unscanned {len(text) - _SCAN_LIMIT} chars omitted]"
    return scanned


def truncate(text: str, limit: int = DEFAULT_MAX_STRING) -> str:
    """Cut ``text`` to ``limit`` characters, saying how much was dropped."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...[truncated {len(text) - limit} chars]"


def redact_string(text: str, limit: int = DEFAULT_MAX_STRING) -> str:
    """Redact secret shapes first, then bound the length.

    The order matters: truncating first could cut a secret in half and leave a
    fragment that no pattern recognises.
    """
    return truncate(redact_text(text), limit)


def redact(
    value: object,
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_string: int = DEFAULT_MAX_STRING,
    max_items: int = DEFAULT_MAX_ITEMS,
) -> object:
    """Return a redacted, size-bounded, JSON-safe copy of ``value``.

    The input is never mutated. Containers beyond ``max_depth`` or ``max_items`` are
    cut with an explicit marker rather than silently dropped.
    """
    return _redact(value, max_depth, max_string, max_items)


def _redact(value: object, depth: int, max_string: int, max_items: int) -> object:
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return redact_string(value, max_string)
    if isinstance(value, bytes | bytearray | memoryview):
        return f"<{len(value)} bytes>"
    if callable(getattr(value, "get_secret_value", None)):
        return REDACTED
    if isinstance(value, Enum):
        return _redact(value.value, depth, max_string, max_items)
    if depth <= 0:
        return "[TRUNCATED]"

    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= max_items:
                result["_truncated"] = f"{len(value) - max_items} more items"
                break
            name = str(key)
            result[name] = (
                REDACTED
                if is_sensitive_key(name)
                else _redact(item, depth - 1, max_string, max_items)
            )
        return result

    if isinstance(value, list | tuple | set | frozenset):
        items = sorted(value, key=repr) if isinstance(value, set | frozenset) else list(value)
        cut = [_redact(item, depth - 1, max_string, max_items) for item in items[:max_items]]
        if len(items) > max_items:
            cut.append(f"[+{len(items) - max_items} more items]")
        return cut

    return redact_string(str(value), max_string)
