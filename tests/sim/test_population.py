import math
from typing import Any

import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError

from civbox.core.canonical import canonical_hash, canonical_json
from civbox.sim.beliefs import validate_beliefs
from civbox.sim.population import (
    MAX_MEMORY_CAPACITY,
    Beta,
    Constant,
    ConstantCount,
    Distribution,
    PopulationProfile,
    Uniform,
    UniformCount,
    sample_initial_beliefs,
    sample_traits,
)
from civbox.sim.rng import RngStreams

from .state_support import make_profile


def rng(seed: int = 1) -> np.random.Generator:
    return RngStreams(seed).generator("population_test")


def profile_with(**overrides: Any) -> PopulationProfile:
    fields: dict[str, Any] = {
        "confirmation_bias": Uniform(low=0.0, high=0.5),
        "susceptibility": Uniform(low=0.0, high=1.0),
        "confidence_bound": Uniform(low=0.2, high=1.0),
        "memory_capacity": UniformCount(low=1, high=5),
        "prior_concentration": 1.0,
    }
    fields.update(overrides)
    return PopulationProfile(**fields)


# --- distributions ---


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (Constant, {"value": True}),
        (Constant, {"value": "0.5"}),
        (Constant, {"value": math.nan}),
        (Constant, {"value": math.inf}),
        (Uniform, {"low": 0.6, "high": 0.4}),
        (Uniform, {"low": 0.0, "high": "1"}),
        (Beta, {"alpha": 0.0, "beta": 1.0}),
        (Beta, {"alpha": 1.0, "beta": -1.0}),
        (Beta, {"alpha": True, "beta": 1.0}),
        (Beta, {"alpha": 1.0, "beta": 1.0, "low": 0.9, "high": 0.1}),
        (ConstantCount, {"value": -1}),
        (ConstantCount, {"value": 2.0}),
        (ConstantCount, {"value": True}),
        (UniformCount, {"low": 5, "high": 2}),
        (UniformCount, {"low": -1, "high": 2}),
    ],
)
def test_invalid_distributions_are_rejected(model: Any, payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Constant(value=0.5, extra=1)  # type: ignore[call-arg]


def test_distributions_are_immutable() -> None:
    with pytest.raises(ValidationError):
        Uniform(low=0.0, high=1.0).low = 0.5  # type: ignore[misc]


def test_a_constant_gives_every_agent_the_same_value() -> None:
    assert Constant(value=0.25).sample(rng(), 50).tolist() == [0.25] * 50
    assert ConstantCount(value=3).sample(rng(), 50).tolist() == [3] * 50


def test_uniform_samples_have_the_right_mean_and_variance_and_stay_in_range() -> None:
    dist, n = Uniform(low=0.2, high=0.8), 40_000
    draws = dist.sample(rng(2), n)
    assert draws.min() >= 0.2
    assert draws.max() <= 0.8
    assert abs(draws.mean() - 0.5) <= 5 * math.sqrt(0.6**2 / 12 / n)
    assert draws.var() == pytest.approx(0.6**2 / 12, rel=0.05)


def test_beta_samples_have_the_right_mean_after_scaling() -> None:
    dist, n = Beta(alpha=2.0, beta=5.0, low=0.1, high=0.9), 40_000
    draws = dist.sample(rng(3), n)
    mean = 0.1 + 0.8 * (2.0 / 7.0)
    sigma = 0.8 * math.sqrt(2 * 5 / (49 * 8)) / math.sqrt(n)
    assert draws.min() >= 0.1
    assert draws.max() <= 0.9
    assert abs(draws.mean() - mean) <= 5 * sigma


def test_beta_samples_never_leave_the_interval_even_for_extreme_shapes() -> None:
    for alpha, beta in ((0.01, 0.01), (500.0, 0.01), (0.01, 500.0), (1e4, 1e4)):
        draws = Beta(alpha=alpha, beta=beta, low=0.0, high=0.9).sample(rng(4), 20_000)
        assert draws.min() >= 0.0
        assert draws.max() <= 0.9
        assert np.isfinite(draws).all()


def test_uniform_counts_cover_both_endpoints_and_nothing_else() -> None:
    draws = UniformCount(low=2, high=6).sample(rng(5), 5000)
    assert draws.dtype == np.int64
    assert set(draws.tolist()) == {2, 3, 4, 5, 6}


def test_a_degenerate_uniform_is_a_constant() -> None:
    assert Uniform(low=0.3, high=0.3).sample(rng(), 10).tolist() == [0.3] * 10
    assert UniformCount(low=4, high=4).sample(rng(), 10).tolist() == [4] * 10


# --- profile validation ---


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("confirmation_bias", Uniform(low=0.0, high=1.0), "confirmation_bias"),
        ("confirmation_bias", Constant(value=1.0), "confirmation_bias"),
        ("confirmation_bias", Uniform(low=-0.1, high=0.5), "confirmation_bias"),
        ("susceptibility", Uniform(low=0.0, high=1.1), "susceptibility"),
        ("susceptibility", Constant(value=-0.01), "susceptibility"),
        ("confidence_bound", Constant(value=0.0), "confidence_bound"),
        ("confidence_bound", Uniform(low=0.5, high=1.01), "confidence_bound"),
        ("memory_capacity", UniformCount(low=1, high=MAX_MEMORY_CAPACITY + 1), "memory_capacity"),
        ("prior_concentration", (1.0,), "at least two"),
    ],
)
def test_traits_outside_their_valid_range_are_rejected(
    field: str, value: Any, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        profile_with(**{field: value})


def test_boundary_values_that_are_allowed_are_accepted() -> None:
    profile_with(
        confirmation_bias=Constant(value=0.0),
        susceptibility=Uniform(low=0.0, high=1.0),
        confidence_bound=Constant(value=1.0),
        memory_capacity=UniformCount(low=0, high=MAX_MEMORY_CAPACITY),
    )


@pytest.mark.parametrize("concentration", [0.0, -1.0, math.nan, True, "1", (1.0, -1.0)])
def test_invalid_prior_concentrations_are_rejected(concentration: Any) -> None:
    with pytest.raises(ValidationError):
        profile_with(prior_concentration=concentration)


def test_every_field_is_required_so_heterogeneity_is_never_defaulted() -> None:
    with pytest.raises(ValidationError):
        PopulationProfile()  # type: ignore[call-arg]


def test_a_profile_round_trips_through_plain_json_data() -> None:
    profile = make_profile(concentration=0.7)
    restored = PopulationProfile.model_validate_json(profile.model_dump_json())
    assert restored == profile
    assert canonical_hash(restored.model_dump(mode="json")) == canonical_hash(
        profile.model_dump(mode="json")
    )


def test_the_distribution_kind_selects_the_class_when_loading_data() -> None:
    adapter = TypeAdapter(Distribution)
    assert isinstance(adapter.validate_python({"kind": "beta", "alpha": 1.0, "beta": 2.0}), Beta)
    assert isinstance(
        adapter.validate_python({"kind": "uniform", "low": 0.0, "high": 1.0}), Uniform
    )
    assert isinstance(adapter.validate_python({"kind": "constant", "value": 0.5}), Constant)
    with pytest.raises(ValidationError):
        adapter.validate_python({"kind": "gaussian", "mean": 0.0})


def test_the_serialised_profile_has_a_pinned_canonical_form() -> None:
    profile = PopulationProfile(
        confirmation_bias=Constant(value=0.25),
        susceptibility=Uniform(low=0.1, high=0.9),
        confidence_bound=Beta(alpha=2.0, beta=3.0, low=0.2, high=1.0),
        memory_capacity=UniformCount(low=2, high=8),
        prior_concentration=(1.0, 2.0, 3.0),
    )
    text = canonical_json(profile.model_dump(mode="json"))
    assert text == (
        '{"confidence_bound":{"alpha":2.0,"beta":3.0,"high":1.0,"kind":"beta","low":0.2},'
        '"confirmation_bias":{"kind":"constant","value":0.25},'
        '"memory_capacity":{"high":8,"kind":"uniform","low":2},'
        '"prior_concentration":[1.0,2.0,3.0],'
        '"susceptibility":{"high":0.9,"kind":"uniform","low":0.1}}'
    )


# --- sampling a population ---


def test_sampled_traits_have_the_documented_shapes_types_and_ranges() -> None:
    traits = sample_traits(make_profile(), 500, RngStreams(1))
    assert traits.n_agents == 500
    for name, array in traits.arrays().items():
        assert array.shape == (500,), name
        assert not array.flags.writeable, name
    assert ((traits.confirmation_bias >= 0.0) & (traits.confirmation_bias < 1.0)).all()
    assert ((traits.susceptibility >= 0.0) & (traits.susceptibility <= 1.0)).all()
    assert ((traits.confidence_bound > 0.0) & (traits.confidence_bound <= 1.0)).all()
    assert traits.memory_capacity.dtype == np.int64
    assert ((traits.memory_capacity >= 0) & (traits.memory_capacity <= 6)).all()


def test_a_population_is_heterogeneous_not_a_set_of_clones() -> None:
    traits = sample_traits(make_profile(), 2000, RngStreams(2))
    for name, array in traits.arrays().items():
        assert array.std() > 0, name
    assert len(set(traits.confirmation_bias.tolist())) > 1900
    assert len(set(traits.susceptibility.tolist())) > 1900
    assert set(traits.memory_capacity.tolist()) == {0, 1, 2, 3, 4, 5, 6}


def test_trait_means_match_their_distributions() -> None:
    n = 40_000
    traits = sample_traits(make_profile(), n, RngStreams(3))
    expected = {
        "confirmation_bias": (0.9 * 2 / 7, 0.9 * math.sqrt(2 * 5 / (49 * 8))),
        "susceptibility": (0.5, 0.8 / math.sqrt(12)),
        "confidence_bound": (0.65, 0.7 / math.sqrt(12)),
        "memory_capacity": (3.0, math.sqrt(((7**2) - 1) / 12)),
    }
    for name, (mean, sd) in expected.items():
        values = traits.arrays()[name]
        assert abs(float(values.mean()) - mean) <= 5 * sd / math.sqrt(n), name


def test_traits_are_statistically_independent_of_each_other() -> None:
    n = 30_000
    arrays = list(sample_traits(make_profile(), n, RngStreams(4)).arrays().values())
    correlations = np.corrcoef(np.stack([a.astype(float) for a in arrays]))
    off_diagonal = correlations[~np.eye(4, dtype=bool)]
    assert np.abs(off_diagonal).max() < 5 / math.sqrt(n)


def test_the_same_seed_gives_the_same_population_and_another_seed_a_different_one() -> None:
    first = sample_traits(make_profile(), 300, RngStreams(5)).arrays()
    again = sample_traits(make_profile(), 300, RngStreams(5)).arrays()
    other = sample_traits(make_profile(), 300, RngStreams(6)).arrays()
    for name in first:
        assert np.array_equal(first[name], again[name])
        assert not np.array_equal(first[name], other[name])


ALTERNATIVES: dict[str, Any] = {
    "confirmation_bias": Beta(alpha=1.0, beta=3.0, low=0.0, high=0.5),
    "susceptibility": Beta(alpha=1.0, beta=1.0),
    "confidence_bound": Beta(alpha=3.0, beta=1.0, low=0.2, high=1.0),
    "memory_capacity": UniformCount(low=0, high=9),
}


@pytest.mark.parametrize("changed", sorted(ALTERNATIVES))
def test_changing_one_trait_distribution_does_not_change_any_other_trait(changed: str) -> None:
    base = sample_traits(profile_with(), 400, RngStreams(7)).arrays()
    different = sample_traits(
        profile_with(**{changed: ALTERNATIVES[changed]}), 400, RngStreams(7)
    ).arrays()
    assert not np.array_equal(base[changed], different[changed])
    for name in base:
        if name != changed:
            assert np.array_equal(base[name], different[name]), f"{name} changed with {changed}"


class AlwaysOne:
    """A stand-in generator whose unit draws are exactly 1.0, the largest possible."""

    def beta(self, a: float, b: float, n: int) -> Any:
        return np.ones(n)

    def random(self, n: int) -> Any:
        return np.ones(n)


@pytest.mark.parametrize(
    "dist",
    [Beta(alpha=1.0, beta=1.0, low=0.407, high=0.947), Uniform(low=0.407, high=0.947)],
)
def test_scaling_can_never_leave_the_declared_interval(dist: Beta | Uniform) -> None:
    # 0.407 + (0.947 - 0.407) * 1.0 exceeds 0.947 by one unit in the last place.
    assert 0.407 + (0.947 - 0.407) * 1.0 > 0.947
    draws = dist.sample(AlwaysOne(), 5)  # type: ignore[arg-type]
    assert draws.max() <= 0.947
    assert draws.min() >= 0.407


def test_a_population_needs_at_least_one_agent() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        sample_traits(make_profile(), 0, RngStreams(1))


# --- initial beliefs ---


def test_initial_beliefs_are_valid_and_unalike() -> None:
    beliefs = sample_initial_beliefs(make_profile(), 500, 4, 1e-9, RngStreams(1))
    validate_beliefs(beliefs, 1e-9)
    assert beliefs.shape == (500, 4)
    assert len({tuple(row) for row in beliefs.tolist()}) == 500


def test_initial_beliefs_average_to_the_normalised_concentration() -> None:
    concentration = (1.0, 2.0, 5.0)
    profile = make_profile()
    profile = PopulationProfile(**{**profile.model_dump(), "prior_concentration": concentration})
    n = 30_000
    beliefs = sample_initial_beliefs(profile, n, 3, 1e-9, RngStreams(2))
    total = sum(concentration)
    for k, alpha in enumerate(concentration):
        mean = alpha / total
        sd = math.sqrt(alpha * (total - alpha) / (total**2 * (total + 1)))
        assert abs(float(beliefs[:, k].mean()) - mean) <= 5 * sd / math.sqrt(n)


def test_small_concentration_gives_confident_agents_large_gives_moderate_ones() -> None:
    confident = sample_initial_beliefs(
        make_profile(concentration=0.05), 2000, 4, 1e-9, RngStreams(3)
    )
    moderate = sample_initial_beliefs(
        make_profile(concentration=50.0), 2000, 4, 1e-9, RngStreams(3)
    )
    assert confident.max(axis=1).mean() > 0.8
    assert moderate.max(axis=1).mean() < 0.5


def test_extremely_small_concentration_still_yields_valid_beliefs() -> None:
    beliefs = sample_initial_beliefs(make_profile(concentration=1e-4), 1000, 5, 1e-9, RngStreams(4))
    assert np.isfinite(beliefs).all()
    validate_beliefs(beliefs, 1e-9)


def test_per_hypothesis_concentration_must_match_the_number_of_hypotheses() -> None:
    profile = profile_with(prior_concentration=(1.0, 2.0, 3.0))
    with pytest.raises(ValueError, match="3 entries for 4 hypotheses"):
        sample_initial_beliefs(profile, 10, 4, 1e-9, RngStreams(1))


def test_initial_beliefs_respect_the_floor_validation() -> None:
    with pytest.raises(ValueError, match="floor"):
        sample_initial_beliefs(make_profile(), 10, 3, 0.5, RngStreams(1))


def test_initial_beliefs_are_deterministic() -> None:
    a = sample_initial_beliefs(make_profile(), 100, 3, 1e-9, RngStreams(9))
    b = sample_initial_beliefs(make_profile(), 100, 3, 1e-9, RngStreams(9))
    assert np.array_equal(a, b)
