"""Social influence: bounded-confidence log-linear pooling (ADR-004).

Agent ``i`` listens to the senders ``j`` it has an edge from, with trust ``T_ij >= 0``.
Each sender's opinion passes a *confidence gate* on its Hellinger distance from the
receiver's own opinion, and the opinions that pass are pooled with the receiver's by a
weighted geometric mean of probabilities, which is a weighted arithmetic mean of
log-probabilities:

    z_i = (1 - s_i) log p_i + s_i * sum_j omega_ij log p_j,
    omega_ij = T_ij g_ij / sum_j' T_ij' g_ij'

where ``s_i`` is the susceptibility and ``g_ij`` the gate. If no sender passes, ``z_i`` is
the receiver's own log-belief unchanged.
"""

from dataclasses import dataclass
from typing import Final, Literal, Self

import numpy as np

from civbox.sim.beliefs import (
    FloatArray,
    IntArray,
    hellinger_distance,
    log_normalize,
    require_array,
)

GateKind = Literal["hard", "soft"]
DEFAULT_GATE_TEMPERATURE: Final = 0.05
_MAX_LOGISTIC_ARGUMENT: Final = 700.0


@dataclass(frozen=True, slots=True)
class InfluenceEdges:
    """Directed, trust-weighted edges, stored sorted by ``(receiver, sender)``.

    Sorting happens once, at construction, so that every accumulation over the edges
    runs in the same order however the edges were supplied. Self-edges are not allowed
    (an agent's own opinion always enters its pool separately) and each
    ``(receiver, sender)`` pair may appear once.
    """

    n_agents: int
    receivers: IntArray
    senders: IntArray
    trust: FloatArray

    @property
    def n_edges(self) -> int:
        return int(self.receivers.shape[0])

    @classmethod
    def from_arrays(
        cls, n_agents: int, receivers: IntArray, senders: IntArray, trust: FloatArray
    ) -> Self:
        for name, array, dtype in (
            ("receivers", receivers, np.int64),
            ("senders", senders, np.int64),
            ("trust", trust, np.float64),
        ):
            require_array(name, array, ndim=1, dtype=dtype)
        count = receivers.shape[0]
        if senders.shape[0] != count or trust.shape[0] != count:
            raise ValueError("receivers, senders and trust must have the same length")
        if count:
            if min(int(receivers.min()), int(senders.min())) < 0 or (
                max(int(receivers.max()), int(senders.max())) >= n_agents
            ):
                raise ValueError(f"edge endpoints must lie in [0, {n_agents})")
            if (receivers == senders).any():
                raise ValueError("self-edges are not allowed")
        if not np.isfinite(trust).all() or (trust < 0.0).any():
            raise ValueError("trust must be finite and non-negative")

        order = np.lexsort((senders, receivers))
        r, s = receivers[order], senders[order]
        if count > 1 and ((r[1:] == r[:-1]) & (s[1:] == s[:-1])).any():
            raise ValueError("each (receiver, sender) pair may appear only once")

        for array in (r, s):
            array.setflags(write=False)
        t = trust[order].copy()
        t.setflags(write=False)
        return cls(n_agents=n_agents, receivers=r, senders=s, trust=t)


def accumulate_by_receiver(edges: InfluenceEdges, values: FloatArray) -> FloatArray:
    """Sum per-edge ``(E, K)`` values into per-receiver ``(N, K)`` totals.

    Edges are added in their stored (sorted) order, one at a time, per hypothesis: the
    sequential accumulation of ``numpy.bincount``, which does not reorder its additions.
    """
    require_array("values", values, ndim=2, dtype=np.float64)
    if values.shape[0] != edges.n_edges:
        raise ValueError("values must have one row per edge")
    totals = np.empty((edges.n_agents, values.shape[1]))
    for k in range(values.shape[1]):
        totals[:, k] = np.bincount(edges.receivers, weights=values[:, k], minlength=edges.n_agents)
    return totals


def gate_weights(
    distance: FloatArray,
    bound: FloatArray,
    kind: GateKind = "hard",
    temperature: float = DEFAULT_GATE_TEMPERATURE,
) -> FloatArray:
    """Confidence gate ``g`` in ``[0, 1]`` for each edge.

    ``hard`` is 1 when ``distance <= bound`` and 0 otherwise. ``soft`` is the logistic
    function ``1 / (1 + exp(-(bound - distance) / temperature))``, a smooth version of the
    same cut-off that is 1/2 at the bound.
    """
    if kind == "hard":
        return (distance <= bound).astype(np.float64)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"temperature must be finite and positive, got {temperature}")
    argument = np.clip(
        (bound - distance) / temperature, -_MAX_LOGISTIC_ARGUMENT, _MAX_LOGISTIC_ARGUMENT
    )
    soft: FloatArray = 1.0 / (1.0 + np.exp(-argument))
    return soft


def pool_beliefs(
    log_beliefs: FloatArray,
    edges: InfluenceEdges,
    susceptibility: FloatArray,
    confidence_bound: FloatArray,
    *,
    gate: GateKind = "hard",
    temperature: float = DEFAULT_GATE_TEMPERATURE,
) -> FloatArray:
    """Pool each agent's normalised log-belief with those of the senders that pass its gate.

    ``log_beliefs`` is ``(N, K)`` and must be normalised (see
    :func:`civbox.sim.beliefs.log_normalize`). ``susceptibility`` is ``(N,)`` in
    ``[0, 1]`` and ``confidence_bound`` is ``(N,)`` in ``(0, 1]``, the largest Hellinger
    distance at which the agent still listens. The gate is evaluated on the beliefs passed
    in, which are everyone's opinions before any pooling in this step. Returns normalised
    log-beliefs.
    """
    require_array("log_beliefs", log_beliefs, ndim=2, dtype=np.float64)
    require_array("susceptibility", susceptibility, ndim=1, dtype=np.float64)
    require_array("confidence_bound", confidence_bound, ndim=1, dtype=np.float64)
    if log_beliefs.shape[0] != edges.n_agents:
        raise ValueError("log_beliefs must have one row per agent in the edge set")
    if susceptibility.shape != (edges.n_agents,) or confidence_bound.shape != (edges.n_agents,):
        raise ValueError("susceptibility and confidence_bound must have one entry per agent")
    if not ((susceptibility >= 0.0).all() and (susceptibility <= 1.0).all()):
        raise ValueError("susceptibility must lie in [0, 1]")
    if not ((confidence_bound > 0.0).all() and (confidence_bound <= 1.0).all()):
        raise ValueError("confidence_bound must lie in (0, 1]")

    probabilities = np.exp(log_beliefs)
    distance = hellinger_distance(probabilities[edges.receivers], probabilities[edges.senders])
    gate_open = gate_weights(distance, confidence_bound[edges.receivers], gate, temperature)
    listening = edges.trust * gate_open

    total = np.bincount(edges.receivers, weights=listening, minlength=edges.n_agents)
    has_listeners = total > 0.0
    # A sender whose weight is zero contributes nothing, so a zero denominator is only
    # ever paired with a zero numerator; the guard avoids the 0/0.
    share = np.divide(
        listening,
        total[edges.receivers],
        out=np.zeros_like(listening),
        where=total[edges.receivers] > 0.0,
    )
    neighbour_mean = accumulate_by_receiver(edges, share[:, None] * log_beliefs[edges.senders])

    effective = susceptibility * has_listeners
    pooled = (1.0 - effective)[:, None] * log_beliefs + effective[:, None] * neighbour_mean
    return log_normalize(pooled)
