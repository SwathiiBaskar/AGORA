import hashlib
from typing import Any

import pytest
from pydantic import ValidationError

from civbox.sim.config import (
    CONFIG_VERSION,
    MAX_POPULATION_SIZE,
    MAX_SEED,
    MAX_SIMULATION_STEPS,
    SimulationConfig,
)

GOLDEN_JSON = (
    '{"config_version":1,"population_size":500,"random_seed":42,'
    '"simulation_mode":"deterministic","simulation_steps":100}'
)
GOLDEN_HASH = "sha256:9d2a50e4290b3659d83cccc3625b9e65d41bdf9306641c593bb86cff77821373"


def make(**overrides: Any) -> SimulationConfig:
    fields: dict[str, Any] = {
        "random_seed": 42,
        "population_size": 500,
        "simulation_steps": 100,
    }
    fields.update(overrides)
    return SimulationConfig(**fields)


def test_canonical_form_and_hash_are_pinned() -> None:
    # Changing the schema or the serialisation changes these values. If this fails
    # deliberately, bump CONFIG_VERSION if existing fields changed meaning.
    config = make()
    assert config.canonical_json() == GOLDEN_JSON
    assert config.config_hash() == GOLDEN_HASH


def test_hash_matches_an_independent_sha256_of_the_json() -> None:
    config = make()
    digest = hashlib.sha256(GOLDEN_JSON.encode("ascii")).hexdigest()
    assert config.config_hash() == f"sha256:{digest}"


def test_defaults_are_part_of_the_canonical_form() -> None:
    config = make()
    assert config.config_version == CONFIG_VERSION == 1
    assert config.simulation_mode == "deterministic"
    assert config.canonical_dict()["simulation_mode"] == "deterministic"


def test_the_hash_does_not_depend_on_the_order_of_fields_in_the_input() -> None:
    reordered = (
        '{"simulation_steps":100,"random_seed":42,"population_size":500,'
        '"simulation_mode":"deterministic","config_version":1}'
    )
    assert SimulationConfig.from_json(reordered).config_hash() == GOLDEN_HASH


def test_a_config_round_trips_through_its_canonical_json() -> None:
    config = make(random_seed=MAX_SEED, population_size=7, simulation_steps=3)
    restored = SimulationConfig.from_json(config.canonical_json())
    assert restored == config
    assert restored.config_hash() == config.config_hash()


@pytest.mark.parametrize(
    "change",
    [
        {"random_seed": 43},
        {"population_size": 501},
        {"simulation_steps": 101},
    ],
)
def test_every_field_affects_the_hash(change: dict[str, int]) -> None:
    assert make(**change).config_hash() != make().config_hash()


@pytest.mark.parametrize(
    ("field", "low", "high"),
    [
        ("random_seed", 0, MAX_SEED),
        ("population_size", 1, MAX_POPULATION_SIZE),
        ("simulation_steps", 1, MAX_SIMULATION_STEPS),
    ],
)
def test_bounds_are_inclusive_and_enforced(field: str, low: int, high: int) -> None:
    make(**{field: low})
    make(**{field: high})
    for outside in (low - 1, high + 1):
        with pytest.raises(ValidationError):
            make(**{field: outside})


def test_the_seed_limit_is_the_largest_integer_a_browser_can_hold_exactly() -> None:
    assert MAX_SEED == 2**53 - 1
    assert float(MAX_SEED) == MAX_SEED
    assert float(MAX_SEED + 1) == float(MAX_SEED + 2)  # precision is already lost


@pytest.mark.parametrize("field", ["random_seed", "population_size", "simulation_steps"])
@pytest.mark.parametrize("bad", [True, "5", 5.0, None])
def test_integers_are_validated_strictly(field: str, bad: object) -> None:
    with pytest.raises(ValidationError):
        make(**{field: bad})


def test_required_fields_have_no_defaults() -> None:
    for field in ("random_seed", "population_size", "simulation_steps"):
        fields = {"random_seed": 1, "population_size": 1, "simulation_steps": 1}
        del fields[field]
        with pytest.raises(ValidationError):
            SimulationConfig.model_validate(fields)


def test_unknown_fields_are_rejected_so_typos_cannot_pass_silently() -> None:
    with pytest.raises(ValidationError):
        make(random_sed=1)


@pytest.mark.parametrize("version", [0, 2, -1])
def test_only_the_supported_version_is_accepted(version: int) -> None:
    with pytest.raises(ValidationError, match="unsupported config_version"):
        make(config_version=version)


def test_only_the_deterministic_mode_is_accepted() -> None:
    with pytest.raises(ValidationError):
        make(simulation_mode="llm_hybrid")


def test_a_config_is_immutable() -> None:
    config = make()
    with pytest.raises(ValidationError):
        config.random_seed = 1  # type: ignore[misc]


def test_malformed_json_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SimulationConfig.from_json("{not json")
    with pytest.raises(ValidationError):
        SimulationConfig.from_json(
            '{"random_seed": 1.5, "population_size": 1, "simulation_steps": 1}'
        )
