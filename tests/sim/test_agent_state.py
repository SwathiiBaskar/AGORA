import itertools
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from civbox.sim.beliefs import floor_beliefs, normalized_entropy
from civbox.sim.evidence import (
    EvidenceBatch,
    event_alignment,
    evidence_weights,
    novelty_discount,
)
from civbox.sim.pooling import InfluenceEdges
from civbox.sim.population import (
    AgentTraits,
    Constant,
    ConstantCount,
    PopulationProfile,
    Uniform,
)
from civbox.sim.rng import RngStreams
from civbox.sim.state import AgentState
from civbox.sim.update import update_beliefs

from .state_support import (
    ReferenceWorld,
    make_profile,
    run_schedule,
    scenario_digest,
    to_arrays,
)

PROPERTY = settings(max_examples=60, deadline=None, derandomize=True, database=None)
REPOSITORY = Path(__file__).resolve().parents[2]
FLOOR = 1e-9


def ints(*v: int) -> Any:
    return np.array(v, dtype=np.int64)


def sampled(
    n_agents: int = 12, n_claims: int = 5, seed: int = 1, profile: PopulationProfile | None = None
) -> AgentState:
    return AgentState.sample(
        profile or make_profile(),
        n_agents=n_agents,
        n_hypotheses=3,
        n_claims=n_claims,
        floor=FLOOR,
        streams=RngStreams(seed),
    )


def snapshot_of(state: AgentState) -> dict[str, np.ndarray]:
    return {name: array.copy() for name, array in state.persistent_arrays().items()}


def unchanged(state: AgentState, before: dict[str, np.ndarray]) -> bool:
    now = state.persistent_arrays()
    return before.keys() == now.keys() and all(np.array_equal(before[k], now[k]) for k in before)


# --- construction ---


def test_a_sampled_state_has_the_requested_shape_and_valid_contents() -> None:
    state = sampled(n_agents=20, n_claims=7)
    assert (state.n_agents, state.n_hypotheses, state.n_claims) == (20, 3, 7)
    assert state.beliefs.shape == (20, 3)
    assert state.committed_tick == -1
    assert not state.available.any()
    assert not state.exposure_counts.any()
    state.validate()


def test_the_same_seed_gives_an_identical_state() -> None:
    a, b = snapshot_of(sampled(seed=4)), snapshot_of(sampled(seed=4))
    assert all(np.array_equal(a[k], b[k]) for k in a)
    c = snapshot_of(sampled(seed=5))
    assert not np.array_equal(a["beliefs"], c["beliefs"])


def test_construction_copies_the_beliefs_it_is_given() -> None:
    source = sampled(n_agents=3).beliefs.copy()
    state = AgentState(sampled(n_agents=3).traits, source, FLOOR, 2)
    original = state.beliefs.copy()
    source[:] = 1.0 / 3.0
    source[0, 0] = 0.5
    assert np.array_equal(state.beliefs, original)


def test_construction_validates_the_beliefs() -> None:
    state = sampled()
    with pytest.raises(ValueError, match="at least the floor"):
        AgentState(state.traits, np.tile([1.0, 0.0, 0.0], (12, 1)), FLOOR, 5)
    with pytest.raises(ValueError, match="one row per agent"):
        AgentState(state.traits, floor_beliefs(np.full((5, 3), 1 / 3), FLOOR), FLOOR, 5)


# --- persistent, derived and transient state ---


def test_the_persistent_arrays_are_exactly_these() -> None:
    assert list(sampled().persistent_arrays()) == [
        "trait_confirmation_bias",
        "trait_susceptibility",
        "trait_confidence_bound",
        "trait_memory_capacity",
        "beliefs",
        "exposure_available",
        "exposure_count",
        "exposure_first_tick",
        "exposure_last_tick",
        "exposure_committed_tick",
        "memory_claims",
        "memory_ticks",
        "memory_written",
    ]


def test_derived_state_is_computed_from_beliefs_and_never_stored() -> None:
    state = sampled(n_agents=3)
    state.replace_beliefs(
        floor_beliefs(np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8], [1 / 3] * 3]), FLOOR)
    )
    derived = state.derived()
    assert derived.leading_hypothesis.tolist() == [0, 2, 0]  # a tie goes to the lowest index
    assert derived.leading_probability[:2].tolist() == pytest.approx([0.7, 0.8])
    assert derived.uncertainty.tolist() == pytest.approx(normalized_entropy(state.beliefs).tolist())
    assert derived.uncertainty[2] == pytest.approx(1.0, abs=1e-12)
    assert not any("uncertainty" in name or "leading" in name for name in state.persistent_arrays())


def test_derived_state_follows_the_beliefs_immediately_and_cannot_change_them() -> None:
    state = sampled(n_agents=2)
    state.replace_beliefs(floor_beliefs(np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05]]), FLOOR))
    first = state.derived()
    state.replace_beliefs(floor_beliefs(np.array([[0.05, 0.05, 0.9], [0.9, 0.05, 0.05]]), FLOOR))
    second = state.derived()
    assert first.leading_hypothesis.tolist() == [0, 1]
    assert second.leading_hypothesis.tolist() == [2, 0]
    before = snapshot_of(state)
    second.leading_probability[:] = 0.0
    second.uncertainty[:] = 9.0
    assert unchanged(state, before)
    assert state.derived().leading_probability is not second.leading_probability


def test_pending_exposures_are_transient_and_outside_the_state_until_committed() -> None:
    state = sampled()
    state.make_available(ints(0, 1), ints(2, 2))
    before = snapshot_of(state)
    buffer = state.new_exposure_buffer()
    buffer.add(ints(0, 1), ints(2, 2))
    assert buffer.n_pending == 2
    assert unchanged(state, before)
    assert not state.was_exposed(0, 2)
    state.commit_exposures(buffer, tick=3)
    assert not unchanged(state, before)
    assert state.was_exposed(0, 2)
    assert buffer.n_pending == 0


def test_committing_an_empty_buffer_only_advances_the_committed_tick() -> None:
    state = sampled()
    before = snapshot_of(state)
    state.commit_exposures(state.new_exposure_buffer(), tick=4)
    assert state.committed_tick == 4
    after = state.persistent_arrays()
    for name in before:
        if name != "exposure_committed_tick":
            assert np.array_equal(before[name], after[name]), name


# --- explicit transitions ---


def test_state_arrays_cannot_be_written_through_the_views_it_hands_out() -> None:
    state = sampled()
    for array in (
        state.beliefs,
        state.available,
        state.exposure_counts,
        state.first_exposure_tick,
        state.last_exposure_tick,
        *state.persistent_arrays().values(),
    ):
        assert not array.flags.writeable


def test_replacing_beliefs_validates_copies_and_is_all_or_nothing() -> None:
    state = sampled(n_agents=4)
    good = floor_beliefs(np.full((4, 3), 1 / 3), FLOOR)
    state.replace_beliefs(good)
    good[0, 0] = 0.0  # mutating the caller's array must not reach the state
    assert state.beliefs[0, 0] == pytest.approx(1 / 3)

    before = snapshot_of(state)
    with pytest.raises(ValueError, match="at least the floor"):
        state.replace_beliefs(np.tile([1.0, 0.0, 0.0], (4, 1)))
    with pytest.raises(ValueError, match="shape"):
        state.replace_beliefs(floor_beliefs(np.full((5, 3), 1 / 3), FLOOR))
    with pytest.raises(ValueError, match="sum to 1"):
        state.replace_beliefs(np.full((4, 3), 0.4))
    assert unchanged(state, before)


def test_a_commit_that_includes_an_unavailable_claim_changes_nothing_and_keeps_the_buffer() -> None:
    state = sampled()
    state.make_available(ints(0), ints(0))
    buffer = state.new_exposure_buffer()
    buffer.add(ints(0), ints(0))  # valid
    buffer.add(ints(1), ints(1))  # never made available
    before = snapshot_of(state)
    with pytest.raises(ValueError, match="agent 1 cannot observe claim 1"):
        state.commit_exposures(buffer, tick=2)
    assert unchanged(state, before)
    assert buffer.n_pending == 2
    assert state.recent_memory(0) == []


def test_a_commit_in_the_past_is_rejected_atomically() -> None:
    state = sampled()
    state.make_available(ints(0), ints(0))
    buffer = state.new_exposure_buffer()
    buffer.add(ints(0), ints(0))
    state.commit_exposures(buffer, tick=5)
    buffer.add(ints(0), ints(0))
    before = snapshot_of(state)
    with pytest.raises(ValueError, match="earlier than"):
        state.commit_exposures(buffer, tick=4)
    assert unchanged(state, before)


def test_a_copy_is_an_independent_snapshot() -> None:
    state = sampled()
    state.make_available(ints(0), ints(0))
    snapshot = state.copy()
    before = snapshot_of(snapshot)
    buffer = state.new_exposure_buffer()
    buffer.add(ints(0), ints(0))
    state.commit_exposures(buffer, tick=1)
    state.replace_beliefs(floor_beliefs(np.full((12, 3), 1 / 3), FLOOR))
    assert unchanged(snapshot, before)
    assert not snapshot.was_exposed(0, 0)
    snapshot.validate()


# --- what an agent was exposed to is not what it remembers ---


def test_an_agent_can_have_been_exposed_to_a_claim_it_no_longer_remembers() -> None:
    profile = PopulationProfile(
        confirmation_bias=Constant(value=0.0),
        susceptibility=Constant(value=0.5),
        confidence_bound=Constant(value=1.0),
        memory_capacity=ConstantCount(value=1),
        prior_concentration=1.0,
    )
    state = sampled(n_agents=2, n_claims=3, profile=profile)
    state.make_available(ints(0, 0), ints(0, 1))
    for tick, claim in enumerate((0, 1)):
        buffer = state.new_exposure_buffer()
        buffer.add(ints(0), ints(claim))
        state.commit_exposures(buffer, tick)

    assert state.was_exposed(0, 0)  # ground truth: it saw claim 0 ...
    assert state.observed_counts(ints(0))[0, 0] == 1.0
    assert state.remembered_counts(ints(0))[0, 0] == 0.0  # ... but has forgotten it
    assert state.remembered_counts(ints(1))[0, 0] == 1.0
    assert state.recent_memory(0) == [(1, 1)]


def test_available_evidence_is_neither_observed_nor_remembered() -> None:
    state = sampled(n_agents=3, n_claims=2)
    state.make_available(ints(0, 1, 2), ints(1, 1, 1))
    assert state.is_available(2, 1)
    assert not state.was_exposed(2, 1)
    assert state.observed_counts(ints(1)).tolist() == [[0.0, 0.0, 0.0]]
    assert state.remembered_counts(ints(1)).tolist() == [[0.0, 0.0, 0.0]]


# --- working with the belief model ---


def test_repeated_exposure_lowers_the_weight_an_agent_gives_the_same_claim() -> None:
    state = sampled(n_agents=40, n_claims=1, profile=make_profile(capacity_low=1, capacity_high=3))
    everyone = np.arange(40, dtype=np.int64)
    state.make_available(everyone, np.zeros(40, dtype=np.int64))
    batch = EvidenceBatch.from_components(
        event_ids=ints(0),
        component_indices=ints(0),
        targets=ints(0),
        signs=np.array([1.0]),
        strengths=np.array([1.0]),
    )
    ones = np.ones((1, 40))
    mean_weights = []
    for tick in range(5):
        novelty = novelty_discount(state.observed_counts(ints(0)), 0.7)
        weights = evidence_weights(
            attention=ones,
            trust=ones,
            bias=state.traits.confirmation_bias,
            alignment=event_alignment(state.beliefs, batch),
            novelty=novelty,
            maximum=3.0,
        )
        mean_weights.append(float(weights.mean()))
        state.replace_beliefs(
            update_beliefs(state.beliefs, floor=FLOOR, batch=batch, weights=weights)
        )
        buffer = state.new_exposure_buffer()
        buffer.add(everyone, np.zeros(40, dtype=np.int64))
        state.commit_exposures(buffer, tick)
    assert all(later < earlier for earlier, later in itertools.pairwise(mean_weights))
    assert mean_weights[0] / mean_weights[-1] > 2.0
    state.validate()


def test_sampled_traits_are_accepted_by_the_belief_model_without_adjustment() -> None:
    state = sampled(n_agents=30)
    ring = np.arange(30, dtype=np.int64)
    edges = InfluenceEdges.from_arrays(30, ring, (ring + 1) % 30, np.ones(30))
    traits: AgentTraits = state.traits
    updated = update_beliefs(
        state.beliefs,
        floor=FLOOR,
        edges=edges,
        susceptibility=traits.susceptibility,
        confidence_bound=traits.confidence_bound,
    )
    state.replace_beliefs(updated)
    state.validate()


# --- agreement with an independent model, across random histories ---


@PROPERTY
@given(
    st.integers(0, 2**32 - 1),
    st.integers(1, 12),
    st.integers(1, 8),
    st.integers(0, 8),
    st.integers(1, 25),
)
def test_the_state_matches_the_reference_model_after_every_tick(
    seed: int, n_agents: int, n_claims: int, capacity_high: int, ticks: int
) -> None:
    state = sampled(n_agents, n_claims, seed, make_profile(capacity_high=capacity_high))
    capacities = [int(c) for c in state.traits.memory_capacity]
    run_schedule(state, ReferenceWorld(capacities, n_claims), seed, ticks)


# --- consistency checks ---


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda s: s._beliefs.__setitem__((0, 0), 0.0), "at least the floor"),
        (lambda s: s._beliefs.__setitem__((0, 0), 0.9), "sum to 1"),
        (lambda s: s._exposure._counts.__setitem__((0, 0), 5), "never available|disagree"),
        (lambda s: s._memory._written.__setitem__(0, -2), "negative"),
        (lambda s: s._memory._capacity.__setitem__(0, 0), "differ from the agents"),
    ],
)
def test_validate_detects_corruption_in_every_component(mutation: Any, message: str) -> None:
    state = sampled(profile=make_profile(capacity_low=2, capacity_high=4))
    mutation(state)
    with pytest.raises(ValueError, match=message):
        state.validate()


# --- determinism ---


def digest_in_fresh_process(seed: int, threads: int | None = None) -> str:
    environment = dict(os.environ)
    if threads is not None:
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            environment[name] = str(threads)
    code = f"from tests.sim.state_support import scenario_digest; print(scenario_digest({seed}))"
    result = subprocess.run(  # noqa: S603 - this interpreter and a fixed program, no shell
        [sys.executable, "-c", code],
        cwd=REPOSITORY,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_a_whole_scenario_is_bit_identical_across_runs_processes_and_thread_counts() -> None:
    expected = scenario_digest(8)
    assert scenario_digest(8) == expected
    assert scenario_digest(9) != expected
    assert digest_in_fresh_process(8) == expected
    assert digest_in_fresh_process(8, threads=1) == expected
    assert digest_in_fresh_process(8, threads=8) == expected


def test_the_pairs_helper_builds_int64_index_arrays() -> None:
    agents, claims = to_arrays([(1, 2), (3, 4)])
    assert agents.dtype == claims.dtype == np.int64
    assert (agents.tolist(), claims.tolist()) == ([1, 3], [2, 4])
    assert Uniform(low=0.0, high=1.0).bounds() == (0.0, 1.0)
