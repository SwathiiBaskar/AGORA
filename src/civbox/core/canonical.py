"""Canonical JSON and content hashing.

Values that are equal as JSON data serialise to identical bytes on every platform
and Python version, so a hash of those bytes identifies the data. Used wherever the
identity of a configuration or a snapshot must be compared by hash.

The canonical form is JSON with sorted keys, no insignificant whitespace, ASCII-only
output (non-ASCII characters are escaped), integers written exactly, and floats
written with Python's shortest round-trip representation. Consequences worth
knowing:

* ``1`` and ``1.0`` are different values and hash differently, as do ``0.0``
  and ``-0.0``.
* ``NaN`` and infinities have no JSON form and are rejected.
* Only ``None``, ``bool``, ``int``, ``float``, ``str``, lists/tuples, and dicts with
  ``str`` keys are accepted. Anything else, such as sets, bytes, or NumPy integer
  and boolean scalars, must be converted explicitly, so a hash never depends on an
  implicit conversion.
"""

import hashlib
import json
import math
from typing import Final

MAX_DEPTH: Final = 64
HASH_PREFIX: Final = "sha256:"


def canonical_json(value: object) -> str:
    """Return the canonical JSON text of ``value``.

    Raises ``TypeError`` for unsupported types or non-string keys, and ``ValueError``
    for non-finite floats or nesting deeper than ``MAX_DEPTH``. The message names the
    location of the offending value.
    """
    _check(value, "$", 0)
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def canonical_hash(value: object) -> str:
    """Return ``sha256:<hex>`` of the canonical JSON of ``value``."""
    digest = hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()
    return f"{HASH_PREFIX}{digest}"


def _check(value: object, path: str, depth: int) -> None:
    if value is None or isinstance(value, bool | int | str):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path}: non-finite float {value!r} has no JSON form")
        return
    if depth >= MAX_DEPTH:
        raise ValueError(f"{path}: nested deeper than {MAX_DEPTH} levels")
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _check(item, f"{path}[{index}]", depth + 1)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path}: object key {key!r} is not a string")
            _check(item, f"{path}.{key}", depth + 1)
        return
    raise TypeError(f"{path}: {type(value).__name__} is not canonical-JSON serialisable")
