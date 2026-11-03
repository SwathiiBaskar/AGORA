"""Named, independent, restorable random-number streams.

Every source of randomness in a simulation draws from its own named stream, so adding,
removing, or reordering one consumer cannot change what any other consumer sees.

Derivation
----------
Stream ``name`` under seed ``s`` is a PCG64 generator seeded with
``SeedSequence(entropy=s, spawn_key=(h(name),))`` where ``h`` is the first 128 bits of
``SHA-256("civbox/rng/v1/" + name)``. The stream depends only on ``(s, name)``: not on
the order streams are created, which other streams exist, or how many numbers they have
produced. ``SeedSequence`` mixes the spawn key into the entropy pool, so distinct names
give statistically independent streams.

Snapshots
---------
:meth:`RngStreams.snapshot` captures the complete state of every stream created so far
as JSON-compatible data. The 128-bit integers of PCG64's state are written as decimal
strings because JSON consumers that parse numbers as doubles (browsers) cannot hold them
exactly. :meth:`RngStreams.from_snapshot` validates a snapshot before using it; NumPy
itself accepts invalid PCG64 states silently, for example an even increment, which
yields a degenerate sequence.
"""

import hashlib
import re
from collections.abc import Mapping
from typing import Final

import numpy as np

_NAME: Final = re.compile(r"[a-z][a-z0-9_]{0,63}")
_DERIVATION_TAG: Final = "civbox/rng/v1"
_BIT_GENERATOR: Final = "PCG64"
_MAX_ENTROPY: Final = 2**128
_SNAPSHOT_KEYS: Final = frozenset({"bit_generator", "state", "increment", "has_uint32", "uinteger"})


def _spawn_key(name: str) -> tuple[int, ...]:
    digest = hashlib.sha256(f"{_DERIVATION_TAG}/{name}".encode("ascii")).digest()
    return (int.from_bytes(digest[:16], "big"),)


class RngStreams:
    """A seeded set of named random streams."""

    def __init__(self, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"seed must be an int, got {type(seed).__name__}")
        if not 0 <= seed < _MAX_ENTROPY:
            raise ValueError(f"seed must satisfy 0 <= seed < 2**128, got {seed}")
        self._seed = seed
        self._streams: dict[str, np.random.Generator] = {}

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def names(self) -> tuple[str, ...]:
        """Names of the streams created so far, sorted."""
        return tuple(sorted(self._streams))

    def generator(self, name: str) -> np.random.Generator:
        """Return the stream called ``name``, creating it on first use.

        Names are lower-case identifiers: a letter followed by up to 63 letters, digits
        or underscores. The same generator object is returned on every call.
        """
        existing = self._streams.get(name)
        if existing is not None:
            return existing
        if _NAME.fullmatch(name) is None:
            raise ValueError(
                f"invalid stream name {name!r}: use lower-case letters, digits and "
                "underscores, starting with a letter, at most 64 characters"
            )
        sequence = np.random.SeedSequence(entropy=self._seed, spawn_key=_spawn_key(name))
        generator = np.random.Generator(np.random.PCG64(sequence))
        self._streams[name] = generator
        return generator

    def snapshot(self) -> dict[str, dict[str, object]]:
        """Complete state of every stream created so far, as JSON-compatible data."""
        return {name: _encode(self._streams[name]) for name in self.names}

    @classmethod
    def from_snapshot(cls, seed: int, snapshot: Mapping[str, object]) -> "RngStreams":
        """Rebuild streams from :meth:`snapshot` output.

        The result continues exactly where the snapshotted instance stopped. Raises
        ``ValueError`` if any entry is malformed or not a valid PCG64 state.
        """
        streams = cls(seed)
        for name, entry in snapshot.items():
            generator = streams.generator(name)
            try:
                generator.bit_generator.state = _decode(entry)
            except ValueError as error:
                raise ValueError(f"stream {name!r}: {error}") from None
        return streams


def _encode(generator: np.random.Generator) -> dict[str, object]:
    state = generator.bit_generator.state
    return {
        "bit_generator": state["bit_generator"],
        "state": str(state["state"]["state"]),
        "increment": str(state["state"]["inc"]),
        "has_uint32": int(state["has_uint32"]),
        "uinteger": int(state["uinteger"]),
    }


def _decode(entry: object) -> dict[str, object]:
    if not isinstance(entry, Mapping):
        raise ValueError("snapshot entry must be an object")
    if set(entry) != _SNAPSHOT_KEYS:
        raise ValueError(f"snapshot entry must have exactly the keys {sorted(_SNAPSHOT_KEYS)}")
    if entry["bit_generator"] != _BIT_GENERATOR:
        raise ValueError(f"unsupported bit generator {entry['bit_generator']!r}")

    state = _decimal(entry["state"], "state", bits=128)
    increment = _decimal(entry["increment"], "increment", bits=128)
    if increment % 2 == 0:
        raise ValueError("increment must be odd")
    has_uint32 = _small_int(entry["has_uint32"], "has_uint32", maximum=1)
    uinteger = _small_int(entry["uinteger"], "uinteger", maximum=2**32 - 1)

    return {
        "bit_generator": _BIT_GENERATOR,
        "state": {"state": state, "inc": increment},
        "has_uint32": has_uint32,
        "uinteger": uinteger,
    }


def _decimal(value: object, field: str, *, bits: int) -> int:
    # isascii() excludes digits from other scripts, which str.isdigit() would accept.
    if not isinstance(value, str) or not value.isascii() or not value.isdigit():
        raise ValueError(f"{field} must be a string of decimal digits")
    if len(value) > len(str(2**bits - 1)):
        raise ValueError(f"{field} must be below 2**{bits}")
    number = int(value)
    if number >= 2**bits:
        raise ValueError(f"{field} must be below 2**{bits}")
    return number


def _small_int(value: object, field: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise ValueError(f"{field} must be an integer between 0 and {maximum}")
    return value
