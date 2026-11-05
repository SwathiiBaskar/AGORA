"""Deterministic scenario builders shared by the belief-model tests."""

import hashlib
from dataclasses import dataclass

import numpy as np

from civbox.sim.beliefs import FloatArray, IntArray, floor_beliefs
from civbox.sim.evidence import EvidenceBatch
from civbox.sim.pooling import GateKind, InfluenceEdges
from civbox.sim.rng import RngStreams
from civbox.sim.update import update_beliefs

MAX_WEIGHT = 3.0


@dataclass(frozen=True)
class RawWorld:
    """A population, its evidence, and its influence edges, before canonical sorting."""

    n_agents: int
    n_hypotheses: int
    floor: float
    beliefs: FloatArray
    event_ids: IntArray
    component_indices: IntArray
    targets: IntArray
    signs: FloatArray
    strengths: FloatArray
    weight_by_event: dict[int, FloatArray]
    receivers: IntArray
    senders: IntArray
    trust: FloatArray
    susceptibility: FloatArray
    confidence_bound: FloatArray
    gate: GateKind

    def batch(self) -> EvidenceBatch:
        return EvidenceBatch.from_components(
            event_ids=self.event_ids,
            component_indices=self.component_indices,
            targets=self.targets,
            signs=self.signs,
            strengths=self.strengths,
        )

    def weights_for(self, batch: EvidenceBatch) -> FloatArray:
        if batch.n_events == 0:
            return np.zeros((0, self.n_agents))
        return np.stack([self.weight_by_event[int(e)] for e in batch.event_ids])

    def edges(self) -> InfluenceEdges:
        return InfluenceEdges.from_arrays(self.n_agents, self.receivers, self.senders, self.trust)

    def update(self) -> FloatArray:
        batch = self.batch()
        return update_beliefs(
            self.beliefs,
            floor=self.floor,
            batch=batch,
            weights=self.weights_for(batch),
            edges=self.edges(),
            susceptibility=self.susceptibility,
            confidence_bound=self.confidence_bound,
            gate=self.gate,
        )


def random_world(
    seed: int,
    *,
    n_agents: int,
    n_hypotheses: int,
    floor: float,
    max_strength: float = 12.0,
    gate: GateKind = "hard",
    concentration: float = 0.3,
) -> RawWorld:
    """A random world; strengths up to ``max_strength`` are large enough to hit the floor."""
    streams = RngStreams(seed)
    rng = streams.generator("belief_world")

    raw = rng.dirichlet(np.full(n_hypotheses, concentration), size=n_agents)
    beliefs = floor_beliefs(raw, floor)

    n_events = int(rng.integers(0, 7))
    ids: list[int] = []
    comps: list[int] = []
    for event in range(n_events):
        for component in range(int(rng.integers(1, 4))):
            ids.append(event * 7 + 3)  # not dense, so canonical ranking is exercised
            comps.append(component)
    n_components = len(ids)
    weight_by_event = {
        event * 7 + 3: rng.random(n_agents) * MAX_WEIGHT for event in range(n_events)
    }

    pairs = [(i, j) for i in range(n_agents) for j in range(n_agents) if i != j]
    keep = rng.random(len(pairs)) < 0.5
    chosen = [pair for pair, k in zip(pairs, keep, strict=True) if k]
    receivers = np.array([p[0] for p in chosen], dtype=np.int64)
    senders = np.array([p[1] for p in chosen], dtype=np.int64)

    return RawWorld(
        n_agents=n_agents,
        n_hypotheses=n_hypotheses,
        floor=floor,
        beliefs=beliefs,
        event_ids=np.array(ids, dtype=np.int64),
        component_indices=np.array(comps, dtype=np.int64),
        targets=rng.integers(0, n_hypotheses, size=n_components).astype(np.int64),
        signs=rng.choice(np.array([-1.0, 1.0]), size=n_components),
        strengths=rng.random(n_components) * max_strength,
        weight_by_event=weight_by_event,
        receivers=receivers,
        senders=senders,
        trust=rng.random(len(chosen)),
        susceptibility=rng.random(n_agents),
        confidence_bound=1.0 - rng.random(n_agents) * 0.9,
        gate=gate,
    )


def reference_update_digest(seed: int) -> str:
    """SHA-256 of the exact bytes of one full update, for cross-process comparison."""
    world = random_world(seed, n_agents=40, n_hypotheses=7, floor=1e-9)
    return hashlib.sha256(np.ascontiguousarray(world.update()).tobytes()).hexdigest()
