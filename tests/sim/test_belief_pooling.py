import math
from typing import Any

import numpy as np
import pytest

from civbox.sim.beliefs import log_normalize
from civbox.sim.pooling import (
    InfluenceEdges,
    accumulate_by_receiver,
    gate_weights,
    pool_beliefs,
)
from civbox.sim.rng import RngStreams


def ints(*v: int) -> Any:
    return np.array(v, dtype=np.int64)


def floats(*v: float) -> Any:
    return np.array(v, dtype=np.float64)


def edges_of(n: int, *edges: tuple[int, int, float]) -> InfluenceEdges:
    """Edges as (receiver, sender, trust)."""
    if not edges:
        return InfluenceEdges.from_arrays(n, ints(), ints(), floats())
    r, s, t = zip(*edges, strict=True)
    return InfluenceEdges.from_arrays(n, ints(*r), ints(*s), floats(*t))


def log_rows(*rows: list[float]) -> Any:
    return np.log(np.array(rows))


# --- edges --------------------------------------------------------------------------


def test_edges_are_stored_sorted_by_receiver_then_sender_whatever_the_input_order() -> None:
    forward = edges_of(4, (0, 1, 0.5), (0, 3, 0.4), (2, 0, 0.9), (3, 1, 0.1))
    scrambled = edges_of(4, (3, 1, 0.1), (0, 3, 0.4), (2, 0, 0.9), (0, 1, 0.5))
    for name in ("receivers", "senders", "trust"):
        assert np.array_equal(getattr(forward, name), getattr(scrambled, name))
    assert forward.receivers.tolist() == [0, 0, 2, 3]
    assert forward.senders.tolist() == [1, 3, 0, 1]
    assert forward.n_edges == 4


def test_edges_are_read_only_and_do_not_alias_their_input() -> None:
    trust = floats(0.5)
    edges = InfluenceEdges.from_arrays(2, ints(0), ints(1), trust)
    trust[0] = 9.0
    assert edges.trust[0] == 0.5
    with pytest.raises(ValueError, match="read-only"):
        edges.receivers[0] = 1


@pytest.mark.parametrize(
    ("edges", "message"),
    [
        (((0, 0, 0.5),), "self-edges"),
        (((0, 1, 0.5), (0, 1, 0.2)), "only once"),
        (((0, 5, 0.5),), "endpoints"),
        (((-1, 1, 0.5),), "endpoints"),
        (((0, 1, -0.5),), "trust"),
        (((0, 1, math.nan),), "trust"),
        (((0, 1, math.inf),), "trust"),
    ],
)
def test_invalid_edges_are_rejected(
    edges: tuple[tuple[int, int, float], ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        edges_of(3, *edges)


def test_edge_arrays_must_have_matching_lengths_and_dtypes() -> None:
    with pytest.raises(ValueError, match="same length"):
        InfluenceEdges.from_arrays(3, ints(0, 1), ints(1), floats(0.5))
    with pytest.raises(TypeError, match="int64"):
        InfluenceEdges.from_arrays(3, np.array([0], dtype=np.int32), ints(1), floats(0.5))


# --- accumulation order --------------------------------------------------------------


def random_edges(seed: int, n: int, count: int) -> InfluenceEdges:
    rng = RngStreams(seed).generator("edges")
    receivers = rng.integers(0, n, size=count)
    senders = (receivers + rng.integers(1, n, size=count)) % n
    pairs = np.unique(np.stack([receivers, senders], axis=1), axis=0)
    return InfluenceEdges.from_arrays(
        n, pairs[:, 0].astype(np.int64), pairs[:, 1].astype(np.int64), rng.random(len(pairs))
    )


def test_accumulation_by_receiver_is_a_strict_left_to_right_sum_in_stored_order() -> None:
    n = 30
    edges = random_edges(1, n, 500)
    rng = RngStreams(2).generator("values")
    values = rng.normal(size=(edges.n_edges, 4)) * 10.0 ** rng.integers(
        -6, 6, size=(edges.n_edges, 4)
    )

    expected = np.zeros((n, 4))
    for receiver in range(n):
        for k in range(4):
            total = 0.0
            for index in np.flatnonzero(edges.receivers == receiver):
                total += float(values[index, k])
            expected[receiver, k] = total
    assert np.array_equal(accumulate_by_receiver(edges, values), expected)


def test_accumulation_gives_zero_to_agents_without_edges() -> None:
    edges = edges_of(4, (0, 1, 1.0))
    totals = accumulate_by_receiver(edges, np.ones((1, 2)))
    assert totals.tolist() == [[1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]


def test_accumulation_with_no_edges_is_all_zero() -> None:
    assert not accumulate_by_receiver(edges_of(3), np.zeros((0, 2))).any()


def test_accumulation_requires_one_row_per_edge() -> None:
    with pytest.raises(ValueError, match="one row per edge"):
        accumulate_by_receiver(edges_of(3, (0, 1, 1.0)), np.zeros((2, 2)))


# --- the gate --------------------------------------------------------------------------


def test_the_hard_gate_is_open_up_to_and_including_the_bound() -> None:
    distance = floats(0.0, 0.2, 0.3, 0.3000001, 1.0)
    assert gate_weights(distance, np.full(5, 0.3)).tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]


def test_a_bound_of_one_never_closes_the_hard_gate() -> None:
    assert (gate_weights(floats(0.0, 0.5, 1.0), np.ones(3)) == 1.0).all()


def test_the_soft_gate_is_one_half_at_the_bound_and_monotone_in_distance() -> None:
    distance = np.linspace(0.0, 1.0, 101)
    soft = gate_weights(distance, np.full(101, 0.4), "soft", 0.05)
    assert gate_weights(floats(0.4), floats(0.4), "soft")[0] == pytest.approx(0.5, abs=1e-15)
    assert (np.diff(soft) <= 0.0).all()
    assert soft[0] > 0.99
    assert soft[-1] < 0.01
    assert ((soft > 0.0) & (soft <= 1.0)).all()


def test_the_soft_gate_approaches_the_hard_gate_as_the_temperature_shrinks() -> None:
    distance = floats(0.1, 0.5)
    sharp = gate_weights(distance, np.full(2, 0.3), "soft", 1e-4)
    assert sharp[0] > 0.999999
    assert sharp[1] < 1e-6


def test_the_soft_gate_survives_extreme_arguments() -> None:
    result = gate_weights(floats(0.0, 1.0), floats(1.0, 1e-9), "soft", 1e-9)
    assert np.isfinite(result).all()


@pytest.mark.parametrize("temperature", [0.0, -1.0, math.nan, math.inf])
def test_the_soft_gate_requires_a_positive_finite_temperature(temperature: float) -> None:
    with pytest.raises(ValueError, match="temperature"):
        gate_weights(floats(0.1), floats(0.5), "soft", temperature)


# --- pooling ----------------------------------------------------------------------------


def pool(
    log_beliefs: Any,
    edges: InfluenceEdges,
    susceptibility: float | list[float] = 1.0,
    bound: float | list[float] = 1.0,
    **kwargs: Any,
) -> Any:
    n = log_beliefs.shape[0]
    return pool_beliefs(
        log_beliefs,
        edges,
        np.broadcast_to(np.asarray(susceptibility, dtype=np.float64), (n,)).copy(),
        np.broadcast_to(np.asarray(bound, dtype=np.float64), (n,)).copy(),
        **kwargs,
    )


def test_two_agents_pooling_with_equal_trust_meet_at_the_normalised_geometric_mean() -> None:
    log_p = log_rows([0.8, 0.2], [0.2, 0.8])
    result = pool(log_p, edges_of(2, (0, 1, 1.0), (1, 0, 1.0)), susceptibility=0.5)
    # Receiver 0: z = 0.5 log p0 + 0.5 log p1 = log sqrt(0.16) per hypothesis: uniform.
    assert np.allclose(np.exp(result), 0.5, rtol=0, atol=1e-15)


def test_full_susceptibility_replaces_an_opinion_with_the_neighbour_mean() -> None:
    log_p = log_rows([0.9, 0.1], [0.3, 0.7])
    result = pool(log_p, edges_of(2, (0, 1, 1.0)), susceptibility=1.0)
    assert np.exp(result[0]).tolist() == pytest.approx([0.3, 0.7], abs=1e-15)
    assert np.exp(result[1]).tolist() == pytest.approx([0.3, 0.7], abs=1e-15)  # no listeners


def test_zero_susceptibility_changes_nothing() -> None:
    log_p = log_normalize(np.random.default_rng(0).normal(size=(4, 3)))
    edges = random_edges(3, 4, 12)
    assert np.allclose(pool(log_p, edges, susceptibility=0.0), log_p, atol=1e-14)


def test_agents_without_listeners_are_unchanged() -> None:
    log_p = log_normalize(np.random.default_rng(1).normal(size=(3, 4)))
    assert np.allclose(pool(log_p, edges_of(3)), log_p, atol=1e-14)


def test_trust_weights_the_influence_of_each_sender() -> None:
    log_p = log_rows([0.5, 0.5], [0.9, 0.1], [0.1, 0.9])
    heavy_on_first = pool(log_p, edges_of(3, (0, 1, 3.0), (0, 2, 1.0)))
    heavy_on_second = pool(log_p, edges_of(3, (0, 1, 1.0), (0, 2, 3.0)))
    assert np.exp(heavy_on_first[0, 0]) > 0.5 > np.exp(heavy_on_second[0, 0])
    # Weights are shares: scaling all trust values changes nothing.
    scaled = pool(log_p, edges_of(3, (0, 1, 30.0), (0, 2, 10.0)))
    assert np.allclose(scaled, heavy_on_first, atol=1e-13)


def test_zero_trust_edges_are_ignored_exactly() -> None:
    log_p = log_rows([0.5, 0.5], [0.9, 0.1])
    assert np.allclose(pool(log_p, edges_of(2, (0, 1, 0.0))), log_p, atol=1e-14)


def test_the_hard_gate_excludes_distant_opinions() -> None:
    log_p = log_rows([0.5, 0.5], [0.52, 0.48], [0.99, 0.01])
    edges = edges_of(3, (0, 1, 1.0), (0, 2, 1.0))
    open_door = pool(log_p, edges, bound=1.0)
    narrow = pool(log_p, edges, bound=0.1)
    only_near = pool(log_p, edges_of(3, (0, 1, 1.0)), bound=1.0)
    assert not np.allclose(open_door[0], narrow[0], atol=1e-3)
    assert np.allclose(narrow[0], only_near[0], atol=1e-14)


def test_each_agent_uses_its_own_confidence_bound() -> None:
    log_p = log_rows([0.5, 0.5], [0.9, 0.1])
    edges = edges_of(2, (0, 1, 1.0), (1, 0, 1.0))
    result = pool(log_p, edges, bound=[1.0, 0.1])
    assert np.exp(result[0, 0]) == pytest.approx(0.9, abs=1e-14)  # listens
    assert np.exp(result[1, 0]) == pytest.approx(0.9, abs=1e-14)  # does not listen


def test_pooling_uses_everyones_opinions_from_before_pooling() -> None:
    # A chain 0 <- 1 <- 2: agent 0 must see agent 1's opinion before agent 1 pools.
    log_p = log_rows([0.1, 0.9], [0.5, 0.5], [0.9, 0.1])
    result = pool(log_p, edges_of(3, (0, 1, 1.0), (1, 2, 1.0)))
    assert np.exp(result[0, 0]) == pytest.approx(0.5, abs=1e-14)
    assert np.exp(result[1, 0]) == pytest.approx(0.9, abs=1e-14)


def test_pooling_returns_normalised_log_beliefs_and_does_not_modify_its_inputs() -> None:
    rng = np.random.default_rng(5)
    log_p = log_normalize(rng.normal(size=(6, 5)) * 3)
    before = log_p.copy()
    result = pool(log_p, random_edges(7, 6, 20), susceptibility=0.7, bound=0.9, gate="soft")
    assert np.abs(np.exp(result).sum(axis=1) - 1.0).max() < 1e-14
    assert np.array_equal(log_p, before)


def test_pooling_validates_its_arguments() -> None:
    log_p = log_rows([0.5, 0.5], [0.5, 0.5])
    edges = edges_of(2, (0, 1, 1.0))
    with pytest.raises(ValueError, match="susceptibility must lie"):
        pool(log_p, edges, susceptibility=1.5)
    with pytest.raises(ValueError, match="confidence_bound must lie"):
        pool(log_p, edges, bound=0.0)
    with pytest.raises(ValueError, match="confidence_bound must lie"):
        pool(log_p, edges, bound=1.5)
    with pytest.raises(ValueError, match="one row per agent"):
        pool(log_rows([0.5, 0.5]), edges)
    with pytest.raises(ValueError, match="one entry per agent"):
        pool_beliefs(log_p, edges, np.ones(3), np.ones(2))
