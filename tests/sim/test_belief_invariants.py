"""The invariants of the belief model (ADR-004), checked over random populations.

Each property draws a seed, population size, number of hypotheses, floor, and gate type,
builds a random world (see ``belief_support``), and checks one invariant. Hypothesis is
run in derandomised mode so a failure reproduces on every run.
"""

import dataclasses
import math
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from civbox.sim.beliefs import (
    FloatArray,
    floor_beliefs,
    hellinger_distance,
    log_normalize,
    normalized_entropy,
    validate_beliefs,
)
from civbox.sim.evidence import EvidenceBatch, accumulate_evidence, evidence_weights
from civbox.sim.pooling import (
    DEFAULT_GATE_TEMPERATURE,
    GateKind,
    InfluenceEdges,
    gate_weights,
    pool_beliefs,
)
from civbox.sim.update import update_beliefs

from .belief_support import RawWorld, random_world

PROPERTY = settings(max_examples=120, deadline=None, derandomize=True, database=None)

seeds = st.integers(0, 2**32 - 1)
n_agents = st.integers(1, 9)
n_hypotheses = st.integers(2, 50)
floor_scales = st.sampled_from([1e-9, 1e-6, 1e-3, 1e-1, 0.5, 0.9])
strengths = st.sampled_from([1.0, 12.0, 5000.0])
gates = st.sampled_from(["hard", "soft"])

TOLERANCE = 1e-9  # in log space; magnitudes reach several hundred


def world_from(
    seed: int, n: int, k: int, scale: float, gate: GateKind = "hard", strength: float = 12.0
) -> RawWorld:
    return random_world(
        seed, n_agents=n, n_hypotheses=k, floor=scale / k, gate=gate, max_strength=strength
    )


@dataclasses.dataclass
class Stages:
    log_after_evidence: FloatArray
    log_after_pooling: FloatArray
    final: FloatArray
    listening: FloatArray
    edges_receivers: Any
    edges_senders: Any


def run_stages(world: RawWorld) -> Stages:
    batch = world.batch()
    edges = world.edges()
    z = accumulate_evidence(np.log(world.beliefs), batch, world.weights_for(batch))
    after_evidence = log_normalize(z)
    after_pooling = pool_beliefs(
        after_evidence, edges, world.susceptibility, world.confidence_bound, gate=world.gate
    )
    probabilities = np.exp(after_evidence)
    distance = hellinger_distance(probabilities[edges.receivers], probabilities[edges.senders])
    open_gate = gate_weights(
        distance, world.confidence_bound[edges.receivers], world.gate, DEFAULT_GATE_TEMPERATURE
    )
    return Stages(
        log_after_evidence=after_evidence,
        log_after_pooling=after_pooling,
        final=floor_beliefs(np.exp(after_pooling), world.floor),
        listening=edges.trust * open_gate,
        edges_receivers=edges.receivers,
        edges_senders=edges.senders,
    )


# --- I1, I2: the floored simplex and finiteness ---


@PROPERTY
@given(seeds, n_agents, n_hypotheses, floor_scales, gates, strengths)
def test_i1_i2_updates_stay_finite_and_on_the_floored_simplex(
    seed: int, n: int, k: int, scale: float, gate: GateKind, strength: float
) -> None:
    world = world_from(seed, n, k, scale, gate, strength)
    out = world.update()
    assert np.isfinite(out).all()
    assert (out >= world.floor).all()
    assert np.abs(out.sum(axis=1) - 1.0).max() <= 1e-12
    validate_beliefs(out, world.floor)
    # The floor bounds every pair of log-odds: the smallest entry is at least the floor,
    # so the largest is at most 1 - (K - 1) * floor.
    saturation = math.log((1.0 - (k - 1) * world.floor) / world.floor)
    log_out = np.log(out)
    assert (log_out.max(axis=1) - log_out.min(axis=1) <= saturation + 1e-9).all()


# --- I3: weights are never negative ---


@PROPERTY
@given(seeds, n_agents, st.integers(1, 6), st.floats(0.1, 5.0))
def test_i3_weights_are_never_negative_and_never_exceed_the_cap(
    seed: int, n: int, events: int, cap: float
) -> None:
    rng = np.random.default_rng(seed)
    weights = evidence_weights(
        attention=rng.random((events, n)),
        trust=rng.random((events, n)),
        bias=rng.random(n) * (1 - 1e-12),
        alignment=rng.random((events, n)) * 2 - 1,
        novelty=rng.random((events, n)),
        maximum=cap,
    )
    assert (weights >= 0.0).all()
    assert (weights <= cap).all()


# --- I4: no evidence, no change ---


@PROPERTY
@given(seeds, n_agents, n_hypotheses, floor_scales)
def test_i4_zero_weights_or_zero_strength_leave_beliefs_unchanged(
    seed: int, n: int, k: int, scale: float
) -> None:
    world = world_from(seed, n, k, scale)
    batch = world.batch()
    zero_weights = np.zeros((batch.n_events, n))
    unchanged = update_beliefs(world.beliefs, floor=world.floor, batch=batch, weights=zero_weights)
    assert np.allclose(unchanged, world.beliefs, rtol=1e-12, atol=0.0)

    silent = EvidenceBatch.from_components(
        event_ids=world.event_ids,
        component_indices=world.component_indices,
        targets=world.targets,
        signs=world.signs,
        strengths=np.zeros_like(world.strengths),
    )
    result = update_beliefs(
        world.beliefs, floor=world.floor, batch=silent, weights=world.weights_for(silent)
    )
    assert np.allclose(result, world.beliefs, rtol=1e-12, atol=0.0)


# --- I5: the order inputs arrive in does not matter ---


@PROPERTY
@given(seeds, n_agents, n_hypotheses, floor_scales, gates, seeds)
def test_i5_results_are_bit_identical_whatever_order_events_and_edges_arrive_in(
    seed: int, n: int, k: int, scale: float, gate: GateKind, shuffle_seed: int
) -> None:
    world = world_from(seed, n, k, scale, gate)
    rng = np.random.default_rng(shuffle_seed)
    components = rng.permutation(len(world.event_ids))
    edge_order = rng.permutation(len(world.receivers))
    shuffled = dataclasses.replace(
        world,
        event_ids=world.event_ids[components],
        component_indices=world.component_indices[components],
        targets=world.targets[components],
        signs=world.signs[components],
        strengths=world.strengths[components],
        receivers=world.receivers[edge_order],
        senders=world.senders[edge_order],
        trust=world.trust[edge_order],
    )
    assert np.array_equal(shuffled.update(), world.update())


# --- I6a, I6b: what pooling and the floor can and cannot do to log ratios ---


@PROPERTY
@given(seeds, n_agents, n_hypotheses, floor_scales, gates, strengths)
def test_i6a_pooling_keeps_every_log_ratio_inside_the_hull_of_its_inputs(
    seed: int, n: int, k: int, scale: float, gate: GateKind, strength: float
) -> None:
    world = world_from(seed, n, k, scale, gate, strength)
    stages = run_stages(world)
    before, after = stages.log_after_evidence, stages.log_after_pooling
    for agent in range(n):
        members = [before[agent]]
        members += [
            before[stages.edges_senders[e]]
            for e in range(len(stages.listening))
            if stages.edges_receivers[e] == agent and stages.listening[e] > 0.0
        ]
        stack = np.stack(members)
        for a in range(k):
            for b in range(k):
                ratios = stack[:, a] - stack[:, b]
                value = after[agent, a] - after[agent, b]
                assert ratios.min() - TOLERANCE <= value <= ratios.max() + TOLERANCE


@PROPERTY
@given(seeds, n_agents, n_hypotheses, floor_scales, gates, strengths)
def test_i6b_the_floor_moves_each_log_ratio_toward_zero_and_never_past_it(
    seed: int, n: int, k: int, scale: float, gate: GateKind, strength: float
) -> None:
    world = world_from(seed, n, k, scale, gate, strength)
    stages = run_stages(world)
    pre = stages.log_after_pooling
    post = np.log(stages.final)
    for a in range(k):
        for b in range(k):
            before = pre[:, a] - pre[:, b]
            after = post[:, a] - post[:, b]
            assert (after >= np.minimum(before, 0.0) - TOLERANCE).all()
            assert (after <= np.maximum(before, 0.0) + TOLERANCE).all()
    untouched = np.exp(pre).min(axis=1) >= world.floor
    assert np.array_equal(stages.final[untouched], np.exp(pre)[untouched])


def test_i6_counterexample_the_final_log_ratio_can_leave_the_hull_of_the_input_log_ratios() -> None:
    # Every agent holds (0.9995, 0.0005) after evidence and trusts everyone. Pooling alone
    # cannot move the log ratio (it is the same for all inputs), but the floor, with
    # eps = 1e-3 above the smaller entry, pulls it toward zero.
    eps = 1e-3
    after_evidence = np.log(np.tile([0.9995, 0.0005], (4, 1)))
    everyone = [(i, j) for i in range(4) for j in range(4) if i != j]
    edges = InfluenceEdges.from_arrays(
        4,
        np.array([i for i, _ in everyone], dtype=np.int64),
        np.array([j for _, j in everyone], dtype=np.int64),
        np.ones(len(everyone)),
    )
    pooled = pool_beliefs(after_evidence, edges, np.ones(4), np.ones(4))
    final = floor_beliefs(np.exp(pooled), eps)

    input_ratio = math.log(0.9995 / 0.0005)  # 7.6004...: every input has this log ratio
    pooled_ratio = float(pooled[0, 0] - pooled[0, 1])
    final_ratio = float(np.log(final[0, 0] / final[0, 1]))

    # The naive claim, that the final ratio stays within the hull of the input ratios
    # (here the single point 7.6004), is false: the floor moved it to 6.9068.
    assert not (input_ratio - 1e-9 <= final_ratio <= input_ratio + 1e-9)
    assert final_ratio == pytest.approx(math.log(0.999 / 0.001), abs=1e-12)
    # I6a: before the floor, pooling kept the ratio exactly inside the hull.
    assert pooled_ratio == pytest.approx(input_ratio, abs=1e-12)
    # I6b: after the floor, the ratio lies between its pre-floor value and zero.
    assert 0.0 < final_ratio < pooled_ratio
    assert final[0, 1] == eps


# --- I7: nothing to listen to, nothing changes ---


@PROPERTY
@given(seeds, n_agents, n_hypotheses, floor_scales, gates, strengths)
def test_i7_pooling_changes_nothing_without_susceptibility_listeners_or_open_gates(
    seed: int, n: int, k: int, scale: float, gate: GateKind, strength: float
) -> None:
    world = world_from(seed, n, k, scale, gate, strength)
    stages = run_stages(world)
    base = stages.log_after_evidence
    edges = world.edges()

    deaf = pool_beliefs(base, edges, np.zeros(n), world.confidence_bound, gate=gate)
    assert np.allclose(deaf, base, rtol=0.0, atol=1e-12)

    zero_trust = dataclasses.replace(world, trust=np.zeros_like(world.trust)).edges()
    ignored = pool_beliefs(
        base, zero_trust, world.susceptibility, world.confidence_bound, gate=gate
    )
    assert np.allclose(ignored, base, rtol=0.0, atol=1e-12)

    if gate == "hard":
        closed = pool_beliefs(base, edges, np.ones(n), np.full(n, 1e-300), gate="hard")
        distinct = ~np.isclose(
            np.exp(base)[edges.receivers], np.exp(base)[edges.senders], rtol=0, atol=1e-6
        ).all(axis=1)
        silent = np.ones(n, dtype=bool)
        silent[np.unique(edges.receivers[~distinct])] = False
        assert np.allclose(closed[silent], base[silent], rtol=0.0, atol=1e-12)


# --- I8: relabelling agents or hypotheses relabels the result ---


@PROPERTY
@given(seeds, n_agents, n_hypotheses, floor_scales, gates, seeds)
def test_i8_relabelling_agents_relabels_the_result(
    seed: int, n: int, k: int, scale: float, gate: GateKind, relabel_seed: int
) -> None:
    world = world_from(seed, n, k, scale, gate)
    pi = np.random.default_rng(relabel_seed).permutation(n)

    def moved(array: Any, axis: int = 0) -> Any:
        out = np.empty_like(array)
        if axis == 0:
            out[pi] = array
        else:
            out[:, pi] = array
        return out

    relabelled = dataclasses.replace(
        world,
        beliefs=moved(world.beliefs),
        weight_by_event={e: moved(w) for e, w in world.weight_by_event.items()},
        receivers=pi[world.receivers].astype(np.int64),
        senders=pi[world.senders].astype(np.int64),
        susceptibility=moved(world.susceptibility),
        confidence_bound=moved(world.confidence_bound),
    )
    assert np.allclose(relabelled.update()[pi], world.update(), rtol=1e-9, atol=0.0)


@PROPERTY
@given(seeds, n_agents, n_hypotheses, floor_scales, gates, seeds)
def test_i8_relabelling_hypotheses_relabels_the_result(
    seed: int, n: int, k: int, scale: float, gate: GateKind, relabel_seed: int
) -> None:
    world = world_from(seed, n, k, scale, gate)
    sigma = np.random.default_rng(relabel_seed).permutation(k)
    beliefs = np.empty_like(world.beliefs)
    beliefs[:, sigma] = world.beliefs
    relabelled = dataclasses.replace(
        world, beliefs=beliefs, targets=sigma[world.targets].astype(np.int64)
    )
    assert np.allclose(relabelled.update()[:, sigma], world.update(), rtol=1e-9, atol=0.0)


# --- I9: determinism and purity ---


@PROPERTY
@given(seeds, n_agents, n_hypotheses, floor_scales, gates)
def test_i9_repeating_an_update_is_bit_identical_and_never_modifies_its_inputs(
    seed: int, n: int, k: int, scale: float, gate: GateKind
) -> None:
    world = world_from(seed, n, k, scale, gate)
    snapshot = (
        world.beliefs.copy(),
        {e: w.copy() for e, w in world.weight_by_event.items()},
        world.trust.copy(),
        world.susceptibility.copy(),
        world.confidence_bound.copy(),
    )
    first, second = world.update(), world.update()
    assert np.array_equal(first, second)
    assert np.array_equal(world.beliefs, snapshot[0])
    assert all(np.array_equal(world.weight_by_event[e], w) for e, w in snapshot[1].items())
    assert np.array_equal(world.trust, snapshot[2])
    assert np.array_equal(world.susceptibility, snapshot[3])
    assert np.array_equal(world.confidence_bound, snapshot[4])


# --- I10: entropy is a fraction ---


@PROPERTY
@given(seeds, st.integers(2, 50), st.sampled_from([0.005, 0.1, 1.0, 20.0]))
def test_i10_normalised_entropy_lies_in_the_unit_interval(seed: int, k: int, alpha: float) -> None:
    rng = np.random.default_rng(seed)
    rows = rng.dirichlet(np.full(k, alpha), size=20)
    rows[0] = np.eye(k)[0]
    rows[1] = 1.0 / k
    entropy = normalized_entropy(rows)
    assert ((entropy >= 0.0) & (entropy <= 1.0)).all()
    assert entropy[0] == 0.0
    assert entropy[1] == pytest.approx(1.0, abs=1e-14)


# --- pooling commutes with evidence that everyone applies identically ---


@PROPERTY
@given(seeds, n_agents, n_hypotheses, floor_scales, strengths)
def test_log_linear_pooling_is_externally_bayesian(
    seed: int, n: int, k: int, scale: float, strength: float
) -> None:
    # If every agent applies the same evidence with the same weight, pooling first and
    # then updating gives the same opinions as updating first and then pooling. (Pooling
    # weights sum to one, so the common likelihood factors out of the geometric mean.)
    world = world_from(seed, n, k, scale, "hard", strength)
    batch = world.batch()
    edges = world.edges()
    per_event = np.random.default_rng(seed).random(batch.n_events) * 3.0
    common = np.tile(per_event[:, None], (1, n))
    ones = np.ones(n)
    start = log_normalize(np.log(world.beliefs))

    pool_then_update = log_normalize(
        accumulate_evidence(pool_beliefs(start, edges, world.susceptibility, ones), batch, common)
    )
    update_then_pool = pool_beliefs(
        log_normalize(accumulate_evidence(start, batch, common)),
        edges,
        world.susceptibility,
        ones,
    )
    assert np.allclose(pool_then_update, update_then_pool, rtol=0.0, atol=1e-9)


# --- the composed update is the stages, applied in order ---


@PROPERTY
@given(seeds, n_agents, n_hypotheses, floor_scales, gates)
def test_the_composed_update_equals_applying_the_stages_in_order(
    seed: int, n: int, k: int, scale: float, gate: GateKind
) -> None:
    world = world_from(seed, n, k, scale, gate)
    assert np.array_equal(run_stages(world).final, world.update())


def test_the_floor_is_applied_once_so_a_resting_population_does_not_drift() -> None:
    eps = 1e-3
    beliefs = floor_beliefs(np.array([[0.9995, 0.0005], [0.3, 0.7]]), eps)
    current = beliefs
    for _ in range(1000):
        current = update_beliefs(current, floor=eps)
    assert np.allclose(current, beliefs, rtol=1e-9, atol=0.0)
    assert current[0, 1] == pytest.approx(eps, rel=1e-9)


# --- argument handling of the composed update ---


def test_update_validates_its_beliefs() -> None:
    with pytest.raises(ValueError, match="at least the floor"):
        update_beliefs(np.array([[1.0, 0.0]]), floor=1e-3)


def test_update_requires_evidence_and_weights_together() -> None:
    world = world_from(1, 3, 3, 1e-6)
    batch = world.batch()
    with pytest.raises(ValueError, match="given together"):
        update_beliefs(world.beliefs, floor=world.floor, batch=batch)
    with pytest.raises(ValueError, match="given together"):
        update_beliefs(world.beliefs, floor=world.floor, weights=world.weights_for(batch))


def test_update_requires_the_pooling_inputs_together() -> None:
    world = world_from(1, 3, 3, 1e-6)
    with pytest.raises(ValueError, match="given together"):
        update_beliefs(world.beliefs, floor=world.floor, edges=world.edges())


def test_update_without_evidence_or_pooling_is_the_identity_up_to_rounding() -> None:
    world = world_from(2, 5, 4, 1e-3)
    assert np.array_equal(
        update_beliefs(world.beliefs, floor=world.floor),
        update_beliefs(world.beliefs, floor=world.floor),
    )
    assert np.allclose(update_beliefs(world.beliefs, floor=world.floor), world.beliefs, rtol=1e-12)


def test_evidence_alone_and_pooling_alone_each_run() -> None:
    world = world_from(3, 5, 4, 1e-6)
    batch = world.batch()
    only_evidence = update_beliefs(
        world.beliefs, floor=world.floor, batch=batch, weights=world.weights_for(batch)
    )
    only_pooling = update_beliefs(
        world.beliefs,
        floor=world.floor,
        edges=world.edges(),
        susceptibility=world.susceptibility,
        confidence_bound=world.confidence_bound,
    )
    for out in (only_evidence, only_pooling):
        validate_beliefs(out, world.floor)
