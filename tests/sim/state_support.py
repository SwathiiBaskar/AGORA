"""Builders and an independent reference model for the agent-state tests."""

import hashlib
from collections import deque

import numpy as np

from civbox.sim.beliefs import IntArray
from civbox.sim.population import (
    Beta,
    PopulationProfile,
    Uniform,
    UniformCount,
)
from civbox.sim.rng import RngStreams
from civbox.sim.state import AgentState


def make_profile(
    *, capacity_low: int = 0, capacity_high: int = 6, concentration: float = 1.0
) -> PopulationProfile:
    return PopulationProfile(
        confirmation_bias=Beta(alpha=2.0, beta=5.0, low=0.0, high=0.9),
        susceptibility=Uniform(low=0.1, high=0.9),
        confidence_bound=Uniform(low=0.3, high=1.0),
        memory_capacity=UniformCount(low=capacity_low, high=capacity_high),
        prior_concentration=concentration,
    )


class ReferenceWorld:
    """What the ledger and memory must contain, computed the slow, obvious way.

    Memory is a ``deque(maxlen=capacity)``, which forgets the oldest record when full. It
    shares no code with the vectorised implementation.
    """

    def __init__(self, capacity: list[int], n_claims: int) -> None:
        self.n_agents = len(capacity)
        self.n_claims = n_claims
        self.capacity = capacity
        self.available: set[tuple[int, int]] = set()
        self.counts: dict[tuple[int, int], int] = {}
        self.first: dict[tuple[int, int], int] = {}
        self.last: dict[tuple[int, int], int] = {}
        self.memory: list[deque[tuple[int, int]]] = [deque(maxlen=c) for c in capacity]
        self.written = [0] * self.n_agents
        self.tick = -1

    def make_available(self, pairs: list[tuple[int, int]]) -> None:
        self.available.update(pairs)

    def commit(self, pairs: list[tuple[int, int]], tick: int) -> None:
        for agent, claim in pairs:
            assert (agent, claim) in self.available
            key = (agent, claim)
            self.counts[key] = self.counts.get(key, 0) + 1
            self.first.setdefault(key, tick)
            self.last[key] = tick
            if self.capacity[agent] > 0:
                self.memory[agent].append((claim, tick))
                self.written[agent] += 1
        self.tick = tick


def assert_matches(state: AgentState, reference: ReferenceWorld) -> None:
    n, m = reference.n_agents, reference.n_claims
    counts = np.zeros((n, m), dtype=np.uint32)
    first = np.full((n, m), -1, dtype=np.int32)
    last = np.full((n, m), -1, dtype=np.int32)
    available = np.zeros((n, m), dtype=np.bool_)
    for (a, c), value in reference.counts.items():
        counts[a, c], first[a, c], last[a, c] = (
            value,
            reference.first[(a, c)],
            reference.last[(a, c)],
        )
    for a, c in reference.available:
        available[a, c] = True

    assert np.array_equal(state.exposure_counts, counts)
    assert np.array_equal(state.first_exposure_tick, first)
    assert np.array_equal(state.last_exposure_tick, last)
    assert np.array_equal(state.available, available)
    assert state.committed_tick == reference.tick
    assert state.persistent_arrays()["memory_written"].tolist() == reference.written
    for agent in range(n):
        assert state.recent_memory(agent) == list(reference.memory[agent])
    claims = np.arange(m, dtype=np.int64)
    expected = np.array(
        [
            [sum(1 for c, _ in reference.memory[a] if c == claim) for a in range(n)]
            for claim in range(m)
        ],
        dtype=np.float64,
    ).reshape(m, n)
    assert np.array_equal(state.remembered_counts(claims), expected)
    state.validate()


def to_arrays(pairs: list[tuple[int, int]]) -> tuple[IntArray, IntArray]:
    agents = np.array([a for a, _ in pairs], dtype=np.int64)
    claims = np.array([c for _, c in pairs], dtype=np.int64)
    return agents, claims


def run_schedule(
    state: AgentState, reference: ReferenceWorld | None, seed: int, ticks: int
) -> None:
    """Drive a random but valid sequence of transitions through the state, and the reference."""
    rng = RngStreams(seed).generator("schedule")
    n, m = state.n_agents, state.n_claims
    tick = 0
    for _ in range(ticks):
        grants = [
            (int(a), int(c))
            for a, c in zip(rng.integers(0, n, 6), rng.integers(0, m, 6), strict=True)
        ]
        agents, claims = to_arrays(grants)
        state.make_available(agents, claims)
        if reference is not None:
            reference.make_available(grants)

        pool = (
            sorted(reference.available)
            if reference is not None
            else sorted((a, c) for a in range(n) for c in range(m) if state.is_available(a, c))
        )
        observed: list[tuple[int, int]] = []
        buffer = state.new_exposure_buffer()
        for _ in range(int(rng.integers(1, 4))):
            if not pool:
                break
            picks = [
                pool[int(i)] for i in rng.integers(0, len(pool), int(rng.integers(0, 3 * n + 1)))
            ]
            observed.extend(picks)
            a, c = to_arrays(picks)
            buffer.add(a, c)
        state.commit_exposures(buffer, tick)
        if reference is not None:
            reference.commit(observed, tick)
            assert_matches(state, reference)
        tick += int(rng.integers(0, 3))  # sometimes the next commit is in the same tick


def scenario_digest(seed: int) -> str:
    """SHA-256 over every persistent array after a deterministic random scenario."""
    streams = RngStreams(seed)
    state = AgentState.sample(
        make_profile(), n_agents=30, n_hypotheses=5, n_claims=12, floor=1e-9, streams=streams
    )
    run_schedule(state, None, seed, ticks=25)
    digest = hashlib.sha256()
    for name, array in state.persistent_arrays().items():
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()
