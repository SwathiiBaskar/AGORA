"""Simulation configuration.

A :class:`SimulationConfig` is the complete, serialisable description of one run.
Two runs with equal configurations, on identical runtimes, produce identical results.

Identity
--------
:meth:`SimulationConfig.config_hash` is the SHA-256 of the configuration's canonical
JSON (see :mod:`civbox.core.canonical`). It covers every field, including defaulted
ones and ``config_version``. A change to the schema that alters the serialisation of
existing fields must increment ``CONFIG_VERSION``.

Bounds
------
Limits exist because configurations are accepted from outside the process. The seed is
capped at 2**53 - 1 so that it survives a round trip through JSON consumers that parse
numbers as IEEE doubles, such as browsers. Integers are validated strictly: booleans,
strings, and floats are not accepted where an integer is required.
"""

from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from civbox.core.canonical import canonical_hash, canonical_json

CONFIG_VERSION: Final = 1
MAX_SEED: Final = 2**53 - 1
MAX_POPULATION_SIZE: Final = 1_000_000
MAX_SIMULATION_STEPS: Final = 1_000_000


class SimulationConfig(BaseModel):
    """Immutable description of a simulation run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    config_version: int = CONFIG_VERSION
    random_seed: Annotated[int, Field(strict=True, ge=0, le=MAX_SEED)]
    population_size: Annotated[int, Field(strict=True, ge=1, le=MAX_POPULATION_SIZE)]
    simulation_steps: Annotated[int, Field(strict=True, ge=1, le=MAX_SIMULATION_STEPS)]
    simulation_mode: Literal["deterministic"] = "deterministic"

    @field_validator("config_version")
    @classmethod
    def _supported_version(cls, value: int) -> int:
        if value != CONFIG_VERSION:
            raise ValueError(
                f"unsupported config_version {value}; this build reads version {CONFIG_VERSION}"
            )
        return value

    def canonical_dict(self) -> dict[str, object]:
        """The configuration as plain JSON-compatible data, the input to the hash."""
        return self.model_dump(mode="json")

    def canonical_json(self) -> str:
        return canonical_json(self.canonical_dict())

    def config_hash(self) -> str:
        """``sha256:<hex>`` identifying this exact configuration."""
        return canonical_hash(self.canonical_dict())

    @classmethod
    def from_json(cls, text: str) -> Self:
        return cls.model_validate_json(text)
