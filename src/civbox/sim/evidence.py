"""Evidence: how observed claims move beliefs (ADR-004).

An *event* is a claim delivered to agents. It consists of one or more *components*, each
a channel observation that points to or rules out one hypothesis with a given strength
``B >= 0``. The components of one event are treated as conditionally independent given the
true hypothesis (a product channel), so their log-likelihoods add.

A component adds ``sign * B`` to the log-belief of its target hypothesis, where ``sign``
is +1 for "points to" and -1 for "rules out". An agent applies an event with a weight
``w >= 0``, which scales every component of that event: weight 1 is exactly Bayes' rule
under the channel, and weight ``w`` is exactly Bayes' rule under the same channel at
strength ``w * B``.
"""

from dataclasses import dataclass
from typing import Self

import numpy as np

from civbox.sim.beliefs import FloatArray, IntArray, require_array


def _readonly[A: np.ndarray](array: A) -> A:
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class EvidenceBatch:
    """The components of the events delivered in one step, in canonical order.

    Events are ordered by ascending event id and components by ascending component
    index within the event, whatever order they were supplied in. ``component_event``
    maps each component to the position of its event in ``event_ids``; per-event arrays
    such as weights are indexed by that position.
    """

    event_ids: IntArray
    component_event: IntArray
    target: IntArray
    sign: FloatArray
    strength: FloatArray

    @property
    def n_events(self) -> int:
        return int(self.event_ids.shape[0])

    @property
    def n_components(self) -> int:
        return int(self.target.shape[0])

    @property
    def signed_strength(self) -> FloatArray:
        """``sign * strength`` for each component."""
        product: FloatArray = self.sign * self.strength
        return product

    @classmethod
    def from_components(
        cls,
        *,
        event_ids: IntArray,
        component_indices: IntArray,
        targets: IntArray,
        signs: FloatArray,
        strengths: FloatArray,
    ) -> Self:
        """Build a batch from components given in any order.

        ``(event_id, component_index)`` pairs must be unique, strengths finite and
        non-negative, and signs exactly +1 or -1. The result is independent of the order
        in which the components were supplied.
        """
        for name, array, dtype in (
            ("event_ids", event_ids, np.int64),
            ("component_indices", component_indices, np.int64),
            ("targets", targets, np.int64),
            ("signs", signs, np.float64),
            ("strengths", strengths, np.float64),
        ):
            require_array(name, array, ndim=1, dtype=dtype)
        count = event_ids.shape[0]
        if not all(a.shape[0] == count for a in (component_indices, targets, signs, strengths)):
            raise ValueError("all component arrays must have the same length")
        if count and (targets < 0).any():
            raise ValueError("targets must be non-negative")
        if not np.isin(signs, (-1.0, 1.0)).all():
            raise ValueError("signs must be exactly +1 or -1")
        if not np.isfinite(strengths).all() or (strengths < 0.0).any():
            raise ValueError("strengths must be finite and non-negative")

        order = np.lexsort((component_indices, event_ids))
        ids, comps = event_ids[order], component_indices[order]
        if count > 1 and ((ids[1:] == ids[:-1]) & (comps[1:] == comps[:-1])).any():
            raise ValueError("(event id, component index) pairs must be unique")

        unique_ids, positions = np.unique(ids, return_inverse=True)
        return cls(
            event_ids=_readonly(unique_ids.astype(np.int64)),
            component_event=_readonly(positions.astype(np.int64)),
            target=_readonly(targets[order].copy()),
            sign=_readonly(signs[order].copy()),
            strength=_readonly(strengths[order].copy()),
        )


def accumulate_evidence(
    log_beliefs: FloatArray, batch: EvidenceBatch, weights: FloatArray
) -> FloatArray:
    """Return ``log p + sum_c w[event(c)] * sign_c * B_c`` onto each target, unnormalised.

    ``log_beliefs`` is ``(N, K)`` and ``weights`` is ``(E, N)`` in the batch's event order.
    Components are added one at a time in canonical order, so the result of the
    floating-point additions is fixed. Pass the result to
    :func:`civbox.sim.beliefs.log_normalize`.
    """
    require_array("log_beliefs", log_beliefs, ndim=2, dtype=np.float64)
    require_array("weights", weights, ndim=2, dtype=np.float64)
    n_agents, n_hypotheses = log_beliefs.shape
    if weights.shape != (batch.n_events, n_agents):
        raise ValueError(
            f"weights must have shape (events, agents) = {(batch.n_events, n_agents)}, "
            f"got {weights.shape}"
        )
    if batch.n_components and int(batch.target.max()) >= n_hypotheses:
        raise ValueError("a component targets a hypothesis outside the belief vectors")

    signed = batch.signed_strength
    accumulated = log_beliefs.copy()
    for c in range(batch.n_components):
        accumulated[:, batch.target[c]] += weights[batch.component_event[c]] * signed[c]
    return accumulated


def component_alignment(beliefs: FloatArray, batch: EvidenceBatch) -> FloatArray:
    """How strongly each component agrees with each agent's current belief, ``(C, N)``.

    ``sign * (2 p_target - 1)``, in ``[-1, 1]``: +1 when the component points to a
    hypothesis the agent is certain of (or rules out one it is certain is false), -1 when
    it contradicts a certainty.
    """
    require_array("beliefs", beliefs, ndim=2, dtype=np.float64)
    alignment = batch.sign[:, None] * (2.0 * beliefs[:, batch.target].T - 1.0)
    clipped: FloatArray = np.clip(alignment, -1.0, 1.0)
    return clipped


def event_alignment(beliefs: FloatArray, batch: EvidenceBatch) -> FloatArray:
    """Mean alignment over the components of each event, ``(E, N)``."""
    per_component = component_alignment(beliefs, batch)
    totals = np.zeros((batch.n_events, beliefs.shape[0]))
    counts = np.zeros(batch.n_events)
    for c in range(batch.n_components):
        event = batch.component_event[c]
        totals[event] += per_component[c]
        counts[event] += 1.0
    result: FloatArray = totals / counts[:, None]
    return result


def novelty_discount(exposure_counts: FloatArray, decay: float) -> FloatArray:
    """``exp(-decay * n)``: the weight given to a claim an agent has seen ``n`` times.

    Treating repeated claims as independent evidence double counts them; this discount
    stands in for the correlation between repeats. ``decay = 0`` disables it.
    """
    if not np.isfinite(decay) or decay < 0.0:
        raise ValueError(f"decay must be finite and non-negative, got {decay}")
    if (exposure_counts < 0.0).any() or not np.isfinite(exposure_counts).all():
        raise ValueError("exposure counts must be finite and non-negative")
    discount: FloatArray = np.exp(-decay * exposure_counts)
    return discount


def evidence_weights(
    *,
    attention: FloatArray,
    trust: FloatArray,
    bias: FloatArray,
    alignment: FloatArray,
    novelty: FloatArray,
    maximum: float,
) -> FloatArray:
    """The weight each agent gives each event: ``min(maximum, a * c * (1 + b * alpha) * nu)``.

    ``attention``, ``trust`` and ``novelty`` are ``(E, N)`` in ``[0, 1]``; ``alignment``
    is ``(E, N)`` in ``[-1, 1]``; ``bias`` (confirmation bias) is ``(N,)`` in ``[0, 1)``,
    which makes ``1 + b * alpha`` positive: bias shrinks the weight of disconfirming
    evidence and enlarges that of confirming evidence but never reverses its direction.
    Inputs outside these ranges are rejected, so the result is always in ``[0, maximum]``.
    """
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError(f"maximum must be finite and positive, got {maximum}")
    for name, array in (("attention", attention), ("trust", trust), ("novelty", novelty)):
        if not (np.isfinite(array).all() and (array >= 0.0).all() and (array <= 1.0).all()):
            raise ValueError(f"{name} must lie in [0, 1]")
    if not (np.isfinite(alignment).all() and (np.abs(alignment) <= 1.0).all()):
        raise ValueError("alignment must lie in [-1, 1]")
    if not (np.isfinite(bias).all() and (bias >= 0.0).all() and (bias < 1.0).all()):
        raise ValueError("bias must lie in [0, 1)")

    raw = attention * trust * (1.0 + bias[None, :] * alignment) * novelty
    capped: FloatArray = np.minimum(maximum, raw)
    return capped
