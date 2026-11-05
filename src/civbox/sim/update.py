"""One synchronous belief update for a population (ADR-004)."""

import numpy as np

from civbox.sim.beliefs import (
    FloatArray,
    floor_beliefs,
    log_normalize,
    validate_beliefs,
)
from civbox.sim.evidence import EvidenceBatch, accumulate_evidence
from civbox.sim.pooling import (
    DEFAULT_GATE_TEMPERATURE,
    GateKind,
    InfluenceEdges,
    pool_beliefs,
)


def update_beliefs(
    beliefs: FloatArray,
    *,
    floor: float,
    batch: EvidenceBatch | None = None,
    weights: FloatArray | None = None,
    edges: InfluenceEdges | None = None,
    susceptibility: FloatArray | None = None,
    confidence_bound: FloatArray | None = None,
    gate: GateKind = "hard",
    temperature: float = DEFAULT_GATE_TEMPERATURE,
) -> FloatArray:
    """Return the beliefs after evidence, then social pooling, then the floor.

    ``beliefs`` is the ``(N, K)`` array on the floored simplex at the start of the step;
    it is not modified. Both stages read only this snapshot: the evidence weights are
    inputs computed from it by the caller, and pooling uses everyone's post-evidence
    opinions, not opinions already changed by pooling. Evidence and pooling operate on
    unfloored log-probabilities and the floor is applied once, at the end.

    Evidence requires ``batch`` and ``weights`` together; pooling requires ``edges``,
    ``susceptibility`` and ``confidence_bound`` together. Either stage may be omitted.
    """
    validate_beliefs(beliefs, floor)
    if (batch is None) != (weights is None):
        raise ValueError("batch and weights must be given together")
    pooling_inputs = (edges, susceptibility, confidence_bound)
    if any(x is None for x in pooling_inputs) and any(x is not None for x in pooling_inputs):
        raise ValueError("edges, susceptibility and confidence_bound must be given together")

    log_beliefs = np.log(beliefs)
    if batch is not None and weights is not None:
        log_beliefs = accumulate_evidence(log_beliefs, batch, weights)
    log_beliefs = log_normalize(log_beliefs)

    if edges is not None and susceptibility is not None and confidence_bound is not None:
        log_beliefs = pool_beliefs(
            log_beliefs,
            edges,
            susceptibility,
            confidence_bound,
            gate=gate,
            temperature=temperature,
        )

    return floor_beliefs(np.exp(log_beliefs), floor)
