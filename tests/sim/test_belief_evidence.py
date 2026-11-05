import math
from typing import Any

import numpy as np
import pytest

from civbox.sim.beliefs import log_normalize
from civbox.sim.evidence import (
    EvidenceBatch,
    accumulate_evidence,
    component_alignment,
    event_alignment,
    evidence_weights,
    novelty_discount,
)
from civbox.sim.rng import RngStreams


def ints(*v: int) -> Any:
    return np.array(v, dtype=np.int64)


def floats(*v: float) -> Any:
    return np.array(v, dtype=np.float64)


def batch_of(*components: tuple[int, int, int, float, float]) -> EvidenceBatch:
    """Components as (event_id, component_index, target, sign, strength)."""
    columns = list(zip(*components, strict=True))
    return EvidenceBatch.from_components(
        event_ids=ints(*columns[0]),
        component_indices=ints(*columns[1]),
        targets=ints(*columns[2]),
        signs=floats(*columns[3]),
        strengths=floats(*columns[4]),
    )


# --- batch construction -------------------------------------------------------


def test_components_are_ordered_by_event_then_component_whatever_the_input_order() -> None:
    forward = batch_of((5, 0, 0, 1.0, 1.0), (5, 1, 2, -1.0, 2.0), (9, 0, 1, 1.0, 3.0))
    scrambled = batch_of((9, 0, 1, 1.0, 3.0), (5, 1, 2, -1.0, 2.0), (5, 0, 0, 1.0, 1.0))
    for name in ("event_ids", "component_event", "target", "sign", "strength"):
        assert np.array_equal(getattr(forward, name), getattr(scrambled, name))
    assert forward.event_ids.tolist() == [5, 9]
    assert forward.component_event.tolist() == [0, 0, 1]
    assert forward.n_events == 2
    assert forward.n_components == 3


def test_event_ids_need_not_be_dense() -> None:
    assert batch_of((1000, 0, 0, 1.0, 1.0), (7, 0, 1, 1.0, 1.0)).event_ids.tolist() == [7, 1000]


def test_a_batch_is_read_only() -> None:
    batch = batch_of((0, 0, 0, 1.0, 1.0))
    with pytest.raises(ValueError, match="read-only"):
        batch.strength[0] = 2.0


def test_the_input_arrays_are_not_aliased() -> None:
    strengths = floats(1.0)
    batch = EvidenceBatch.from_components(
        event_ids=ints(0),
        component_indices=ints(0),
        targets=ints(0),
        signs=floats(1.0),
        strengths=strengths,
    )
    strengths[0] = 99.0
    assert batch.strength[0] == 1.0


def test_an_empty_batch_is_valid() -> None:
    batch = EvidenceBatch.from_components(
        event_ids=ints(),
        component_indices=ints(),
        targets=ints(),
        signs=floats(),
        strengths=floats(),
    )
    assert (batch.n_events, batch.n_components) == (0, 0)
    log_p = np.log(np.full((3, 2), 0.5))
    result = accumulate_evidence(log_p, batch, np.zeros((0, 3)))
    assert np.array_equal(result, log_p)
    assert result is not log_p


@pytest.mark.parametrize(
    ("components", "message"),
    [
        (((0, 0, 0, 1.0, 1.0), (0, 0, 1, 1.0, 1.0)), "must be unique"),
        (((0, 0, -1, 1.0, 1.0),), "non-negative"),
        (((0, 0, 0, 0.0, 1.0),), "exactly \\+1 or -1"),
        (((0, 0, 0, 2.0, 1.0),), "exactly \\+1 or -1"),
        (((0, 0, 0, 1.0, -0.5),), "finite and non-negative"),
        (((0, 0, 0, 1.0, math.nan),), "finite and non-negative"),
        (((0, 0, 0, 1.0, math.inf),), "finite and non-negative"),
    ],
)
def test_invalid_components_are_rejected(
    components: tuple[tuple[int, int, int, float, float], ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        batch_of(*components)


def test_mismatched_lengths_and_wrong_dtypes_are_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        EvidenceBatch.from_components(
            event_ids=ints(0, 1),
            component_indices=ints(0),
            targets=ints(0),
            signs=floats(1.0),
            strengths=floats(1.0),
        )
    with pytest.raises(TypeError, match="int64"):
        EvidenceBatch.from_components(
            event_ids=np.array([0], dtype=np.int32),
            component_indices=ints(0),
            targets=ints(0),
            signs=floats(1.0),
            strengths=floats(1.0),
        )


# --- accumulation -------------------------------------------------------------


def test_a_component_adds_its_signed_weighted_strength_to_its_target_only() -> None:
    batch = batch_of((0, 0, 1, -1.0, 2.0))
    log_p = np.log(np.full((2, 3), 1 / 3))
    weights = np.array([[1.0, 0.5]])
    result = accumulate_evidence(log_p, batch, weights)
    assert result[0, 1] == pytest.approx(log_p[0, 1] - 2.0, abs=1e-15)
    assert result[1, 1] == pytest.approx(log_p[1, 1] - 1.0, abs=1e-15)
    assert np.array_equal(result[:, [0, 2]], log_p[:, [0, 2]])


def test_accumulation_does_not_modify_its_inputs() -> None:
    batch = batch_of((0, 0, 0, 1.0, 1.0))
    log_p = np.log(np.full((2, 2), 0.5))
    weights = np.ones((1, 2))
    log_before, weights_before = log_p.copy(), weights.copy()
    accumulate_evidence(log_p, batch, weights)
    assert np.array_equal(log_p, log_before)
    assert np.array_equal(weights, weights_before)


def test_accumulation_adds_components_strictly_in_canonical_order() -> None:
    rng = RngStreams(21).generator("evidence")
    n_agents, n_hyp, n_comp = 25, 4, 60
    batch = EvidenceBatch.from_components(
        event_ids=np.arange(n_comp, dtype=np.int64) // 3,
        component_indices=np.arange(n_comp, dtype=np.int64) % 3,
        targets=rng.integers(0, n_hyp, size=n_comp).astype(np.int64),
        signs=rng.choice(np.array([-1.0, 1.0]), size=n_comp),
        strengths=rng.random(n_comp) * 10.0 ** rng.integers(-3, 4, size=n_comp),
    )
    weights = rng.random((batch.n_events, n_agents)) * 3.0
    log_p = rng.normal(size=(n_agents, n_hyp)) * 10.0 ** rng.integers(-2, 3, size=(n_agents, n_hyp))

    expected = log_p.copy()
    for c in range(n_comp):
        for i in range(n_agents):
            term = float(weights[batch.component_event[c], i]) * float(
                batch.sign[c] * batch.strength[c]
            )
            expected[i, batch.target[c]] = float(expected[i, batch.target[c]]) + term
    assert np.array_equal(accumulate_evidence(log_p, batch, weights), expected)


def test_accumulation_rejects_inconsistent_shapes() -> None:
    batch = batch_of((0, 0, 3, 1.0, 1.0))
    with pytest.raises(ValueError, match="outside the belief vectors"):
        accumulate_evidence(np.zeros((2, 3)), batch, np.ones((1, 2)))
    with pytest.raises(ValueError, match="weights must have shape"):
        accumulate_evidence(np.zeros((2, 4)), batch, np.ones((2, 2)))


# --- alignment, novelty, weights ----------------------------------------------------


def test_alignment_is_plus_one_for_certain_agreement_and_minus_one_for_certain_contradiction() -> (
    None
):
    beliefs = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    points_to_first = batch_of((0, 0, 0, 1.0, 1.0))
    rules_out_first = batch_of((0, 0, 0, -1.0, 1.0))
    assert component_alignment(beliefs, points_to_first)[0].tolist() == [1.0, -1.0, 0.0]
    assert component_alignment(beliefs, rules_out_first)[0].tolist() == [-1.0, 1.0, -0.0]


def test_alignment_is_clipped_into_the_unit_interval() -> None:
    beliefs = np.array([[1.0 + 4e-16, -4e-16]])
    alignment = component_alignment(beliefs, batch_of((0, 0, 0, 1.0, 1.0), (1, 0, 1, 1.0, 1.0)))
    assert (np.abs(alignment) <= 1.0).all()


def test_event_alignment_is_the_mean_over_the_components_of_each_event() -> None:
    beliefs = np.array([[0.9, 0.1]])
    batch = batch_of((0, 0, 0, 1.0, 1.0), (0, 1, 1, 1.0, 1.0), (1, 0, 0, 1.0, 1.0))
    result = event_alignment(beliefs, batch)
    assert result[0, 0] == pytest.approx((0.8 + -0.8) / 2, abs=1e-15)
    assert result[1, 0] == pytest.approx(0.8, abs=1e-15)


def test_novelty_discount_decays_with_exposure_and_is_disabled_by_zero_decay() -> None:
    counts = np.array([0.0, 1.0, 2.0, 10.0])
    assert novelty_discount(counts, 0.0).tolist() == [1.0, 1.0, 1.0, 1.0]
    result = novelty_discount(counts, 0.5)
    assert result[0] == 1.0
    assert np.allclose(result, np.exp(-0.5 * counts), rtol=1e-15)
    assert (np.diff(result) < 0).all()


@pytest.mark.parametrize(
    ("counts", "decay"), [([-1.0], 0.5), ([math.nan], 0.5), ([1.0], -0.1), ([1.0], math.inf)]
)
def test_novelty_discount_rejects_invalid_inputs(counts: list[float], decay: float) -> None:
    with pytest.raises(ValueError, match=r"exposure counts|decay"):
        novelty_discount(np.array(counts), decay)


def weights_for(**overrides: Any) -> Any:
    ones = np.ones((2, 3))
    arguments: dict[str, Any] = {
        "attention": ones,
        "trust": ones,
        "bias": np.zeros(3),
        "alignment": np.zeros((2, 3)),
        "novelty": ones,
        "maximum": 2.0,
    }
    arguments.update(overrides)
    return evidence_weights(**arguments)


def test_the_weight_is_the_product_of_its_factors() -> None:
    result = weights_for(
        attention=np.full((2, 3), 0.5),
        trust=np.full((2, 3), 0.8),
        bias=np.full(3, 0.5),
        alignment=np.full((2, 3), 1.0),
        novelty=np.full((2, 3), 0.25),
        maximum=10.0,
    )
    assert np.allclose(result, 0.5 * 0.8 * (1 + 0.5 * 1.0) * 0.25, rtol=1e-15)


def test_confirmation_bias_shrinks_disconfirming_and_enlarges_confirming_evidence() -> None:
    alignment = np.array([[-1.0, 0.0, 1.0]] * 2)
    result = weights_for(bias=np.full(3, 0.6), alignment=alignment, maximum=10.0)
    assert result[0].tolist() == pytest.approx([0.4, 1.0, 1.6])


def test_the_weight_is_capped() -> None:
    assert weights_for(bias=np.full(3, 0.9), alignment=np.ones((2, 3)), maximum=1.5).max() == 1.5


def test_zero_attention_trust_or_novelty_gives_exactly_zero_weight() -> None:
    for name in ("attention", "trust", "novelty"):
        assert (weights_for(**{name: np.zeros((2, 3))}) == 0.0).all()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"attention": np.full((2, 3), 1.5)}, "attention"),
        ({"trust": np.full((2, 3), -0.1)}, "trust"),
        ({"novelty": np.full((2, 3), math.nan)}, "novelty"),
        ({"alignment": np.full((2, 3), 1.5)}, "alignment"),
        ({"bias": np.full(3, 1.0)}, "bias"),
        ({"bias": np.full(3, -0.1)}, "bias"),
        ({"maximum": 0.0}, "maximum"),
        ({"maximum": math.inf}, "maximum"),
    ],
)
def test_out_of_range_inputs_are_rejected_so_weights_can_never_be_negative(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        weights_for(**overrides)


def test_normalising_after_accumulating_gives_a_probability_vector() -> None:
    batch = batch_of((0, 0, 0, 1.0, 5.0))
    log_p = np.log(np.array([[0.2, 0.8]]))
    z = accumulate_evidence(log_p, batch, np.ones((1, 1)))
    assert np.exp(log_normalize(z)).sum() == pytest.approx(1.0, abs=1e-15)
