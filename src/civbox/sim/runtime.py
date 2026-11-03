"""Fingerprint of the numerical runtime a result was produced on.

Floating-point results are only guaranteed identical when the runtime is identical. In
particular NumPy selects among several implementations of ``exp``, ``log`` and other
functions at import time according to the CPU's instruction-set extensions, and those
implementations do not always agree to the last bit. A :class:`RuntimeFingerprint`
records what is needed to tell whether two results were produced under the same
conditions.

CPU dispatch
------------
NumPy exposes *dispatch targets* (instruction-set levels it has optimised code for) and
the subset that is active in this process. The environment variable
``NPY_DISABLE_CPU_FEATURES``, read when NumPy is first imported, removes targets from the
active set. Setting it to every available target restricts NumPy to its baseline code
path, which is identical on all CPUs that NumPy build supports. The variable is
case-sensitive and an unrecognised name is ignored with only an ``ImportWarning``, so
the fingerprint reports what is actually active rather than what was requested.
Individual feature flags such as ``AVX2`` keep reporting the hardware's capability after
a target is disabled; only the targets reflect the restriction, which is why they are
what is recorded.

The introspection relies on attributes of a private NumPy module. If they are missing,
``cpu_introspection`` is False and the fingerprint makes no claim about dispatch.
"""

import importlib
import os
import platform
import sys
from typing import Final, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, StrictBool

DISPATCH_ENVIRONMENT_VARIABLE: Final = "NPY_DISABLE_CPU_FEATURES"
_NUMPY_INTERNALS: Final = "numpy._core._multiarray_umath"


class RuntimeFingerprint(BaseModel):
    """What the numerical results of this process depend on."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    python_implementation: str
    python_version: str
    system: str
    machine: str
    byte_order: str
    numpy_version: str
    cpu_introspection: StrictBool
    cpu_baseline: tuple[str, ...]
    cpu_dispatch_available: tuple[str, ...]
    cpu_dispatch_active: tuple[str, ...]
    npy_disable_cpu_features: str | None

    @classmethod
    def collect(cls) -> Self:
        """Describe the current process."""
        introspection, baseline, available, active = _read_cpu_dispatch()
        return cls(
            python_implementation=platform.python_implementation(),
            python_version=platform.python_version(),
            system=platform.system(),
            machine=platform.machine(),
            byte_order=sys.byteorder,
            numpy_version=np.__version__,
            cpu_introspection=introspection,
            cpu_baseline=baseline,
            cpu_dispatch_available=available,
            cpu_dispatch_active=active,
            npy_disable_cpu_features=os.environ.get(DISPATCH_ENVIRONMENT_VARIABLE) or None,
        )

    @property
    def restricted_to_baseline(self) -> bool:
        """True if NumPy is confirmed to be running only its baseline code path.

        False when any dispatch target is active, and also when the dispatch state could
        not be read, because nothing can then be vouched for.
        """
        return self.cpu_introspection and not self.cpu_dispatch_active

    @property
    def baseline_only_setting(self) -> str:
        """Value for ``NPY_DISABLE_CPU_FEATURES`` that restricts this NumPy build to baseline."""
        return " ".join(self.cpu_dispatch_available)

    def to_dict(self) -> dict[str, object]:
        """JSON-compatible form, suitable for storing next to results."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: object) -> Self:
        return cls.model_validate(data)

    def differences(self, other: Self) -> tuple[str, ...]:
        """Names of the fields whose values differ from ``other``'s, in field order."""
        return tuple(
            name for name in type(self).model_fields if getattr(self, name) != getattr(other, name)
        )


def _read_cpu_dispatch() -> tuple[bool, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    try:
        internals = importlib.import_module(_NUMPY_INTERNALS)
        baseline = tuple(str(name) for name in internals.__cpu_baseline__)
        available = tuple(str(name) for name in internals.__cpu_dispatch__)
        features = internals.__cpu_features__
        active = tuple(name for name in available if features.get(name))
    except (ImportError, AttributeError, TypeError):
        return False, (), (), ()
    return True, baseline, available, active
