"""Belief representation and the quantities defined on it (ADR-004).

An agent's belief about K mutually exclusive hypotheses is a probability vector
``p`` on the *floored simplex*: ``p_k >= floor`` for every k and ``sum(p) == 1``. The
floor ``eps`` is strictly between 0 and ``1/K``. Beliefs for N agents are an ``(N, K)``
float64 array, one row per agent.

All reductions over the hypothesis axis are performed strictly left to right by
:func:`sum_over_hypotheses`, so their rounding does not depend on NumPy's choice of
summation strategy.
"""

import math
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

DEFAULT_BELIEF_FLOOR: Final = 1e-9
MIN_HYPOTHESES: Final = 2

# Tolerances for accepting externally supplied beliefs. Beliefs produced by this module
# satisfy the floor exactly and sum to one to about 1e-15.
_FLOOR_TOLERANCE: Final = 1e-12
_SUM_TOLERANCE: Final = 1e-9


def require_array(
    name: str,
    array: object,
    *,
    ndim: int,
    dtype: type[np.floating] | type[np.integer],
) -> None:
    """Raise unless ``array`` is an ndarray of the given dtype and dimensionality."""
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(array).__name__}")
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {np.dtype(dtype)}, got {array.dtype}")
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional, got {array.ndim} dimensions")


def check_floor(n_hypotheses: int, floor: float) -> None:
    """Raise unless there are at least two hypotheses and ``0 < floor < 1/K``."""
    if n_hypotheses < MIN_HYPOTHESES:
        raise ValueError(f"at least {MIN_HYPOTHESES} hypotheses are required, got {n_hypotheses}")
    if not 0.0 < floor < 1.0 / n_hypotheses:
        raise ValueError(f"floor must satisfy 0 < floor < 1/K = {1.0 / n_hypotheses}, got {floor}")


def sum_over_hypotheses(values: FloatArray) -> FloatArray:
    """Sum along the last axis strictly left to right.

    Each addition is a separate elementwise operation, so the result does not depend on
    how NumPy would otherwise block, unroll, or vectorise a reduction.
    """
    total = values[..., 0].copy()
    for k in range(1, values.shape[-1]):
        total += values[..., k]
    return total


def validate_beliefs(beliefs: FloatArray, floor: float) -> None:
    """Raise unless ``beliefs`` is an ``(N, K)`` array on the floored simplex."""
    require_array("beliefs", beliefs, ndim=2, dtype=np.float64)
    check_floor(beliefs.shape[1], floor)
    if not np.isfinite(beliefs).all():
        raise ValueError("beliefs must be finite")
    if (beliefs < floor * (1.0 - _FLOOR_TOLERANCE)).any():
        raise ValueError(f"beliefs must be at least the floor {floor}")
    if (np.abs(sum_over_hypotheses(beliefs) - 1.0) > _SUM_TOLERANCE).any():
        raise ValueError("each row of beliefs must sum to 1")


def floor_beliefs(probabilities: FloatArray, floor: float) -> FloatArray:
    """Project probability rows onto the floored simplex by minimal uniform mixing.

    A row whose smallest entry is already at least ``floor`` is returned unchanged,
    bit for bit. Otherwise it is mixed with the uniform distribution,
    ``(1 - m) q + m / K``, using the smallest mixing weight ``m`` that brings the
    smallest entry exactly to ``floor``. Consequently the map is idempotent, preserves
    the order of the entries, and moves any pair of entries toward each other: the log
    ratio of two entries after flooring lies between their log ratio before and zero.

    Rows must be non-negative and sum to one.
    """
    require_array("probabilities", probabilities, ndim=2, dtype=np.float64)
    n_hypotheses = probabilities.shape[1]
    check_floor(n_hypotheses, floor)

    smallest = probabilities.min(axis=1)
    mixing = np.zeros_like(smallest)
    binding = smallest < floor
    mixing[binding] = (floor - smallest[binding]) / (1.0 / n_hypotheses - smallest[binding])

    mixed = (1.0 - mixing)[:, None] * probabilities + (mixing / n_hypotheses)[:, None]
    # Rounding can leave the smallest entry a few units in the last place below the
    # floor; raising it changes a row sum by far less than float64 can resolve.
    floored: FloatArray = np.maximum(mixed, floor)
    return floored


def log_normalize(unnormalized_log: FloatArray) -> FloatArray:
    """Normalise log-weights so that the exponentials of each row sum to one.

    Returns ``z - log(sum(exp(z)))`` per row, computed with the row maximum subtracted
    first so that nothing overflows.
    """
    require_array("unnormalized_log", unnormalized_log, ndim=2, dtype=np.float64)
    shifted = unnormalized_log - unnormalized_log.max(axis=1, keepdims=True)
    total = sum_over_hypotheses(np.exp(shifted))
    return shifted - np.log(total)[:, None]


def normalized_entropy(probabilities: FloatArray) -> FloatArray:
    """Shannon entropy of each row divided by ``log K``, so in ``[0, 1]``.

    Zero entries contribute nothing (``0 log 0 = 0``). 1 is the uniform distribution and
    0 is certainty.
    """
    require_array("probabilities", probabilities, ndim=2, dtype=np.float64)
    positive = probabilities > 0.0
    terms = np.where(positive, probabilities * np.log(np.where(positive, probabilities, 1.0)), 0.0)
    entropy = -sum_over_hypotheses(terms) / math.log(probabilities.shape[1])
    return np.clip(entropy + 0.0, 0.0, 1.0)


def hellinger_distance(p: FloatArray, q: FloatArray) -> FloatArray:
    """Hellinger distance between corresponding rows of ``p`` and ``q``, in ``[0, 1]``.

    ``sqrt(1 - sum_k sqrt(p_k q_k))``. It is symmetric, bounded, and defined when entries
    are zero.
    """
    overlap = sum_over_hypotheses(np.sqrt(p * q))
    return np.sqrt(np.clip(1.0 - overlap, 0.0, 1.0))


ChannelKind = Literal["point_to", "rules_out"]


def _check_channel(n_hypotheses: int, strength: float) -> None:
    if n_hypotheses < MIN_HYPOTHESES:
        raise ValueError(f"at least {MIN_HYPOTHESES} hypotheses are required")
    if not math.isfinite(strength) or strength < 0.0:
        raise ValueError(f"strength must be finite and non-negative, got {strength}")


def point_to_channel(strength: float, n_hypotheses: int) -> FloatArray:
    """Row-stochastic ``(K, K)`` matrix ``C[h, y]`` of the K-ary symmetric channel.

    Given true hypothesis ``h`` the source reports ``y = h`` with probability
    ``e^B / (e^B + K - 1)`` and each other value with equal probability. Observing
    ``y = m`` multiplies the likelihood of hypothesis ``m`` relative to every other
    hypothesis by exactly ``e^B``, where ``B`` is ``strength``.
    """
    _check_channel(n_hypotheses, strength)
    decay = math.exp(-strength)
    denominator = 1.0 + (n_hypotheses - 1) * decay
    matrix = np.full((n_hypotheses, n_hypotheses), decay / denominator)
    np.fill_diagonal(matrix, 1.0 / denominator)
    return matrix


def rules_out_channel(target: int, strength: float, n_hypotheses: int) -> FloatArray:
    """Row-stochastic ``(K, 2)`` matrix ``C[h, y]`` of the binary rule-out channel.

    Column 0 is the outcome "rules out ``target``", column 1 is "does not rule it out".
    Under hypothesis ``target`` the first outcome has probability ``1 / (1 + e^B)``;
    under every other hypothesis it has probability ``e^B / (1 + e^B)``. Observing the
    first outcome multiplies the likelihood of ``target`` by ``e^-B`` relative to every
    other hypothesis; observing the second multiplies it by ``e^B``.
    """
    _check_channel(n_hypotheses, strength)
    if not 0 <= target < n_hypotheses:
        raise ValueError(f"target must be in [0, {n_hypotheses}), got {target}")
    decay = math.exp(-strength)
    denominator = 1.0 + decay
    high, low = 1.0 / denominator, decay / denominator
    matrix = np.empty((n_hypotheses, 2))
    matrix[:, 0] = high
    matrix[:, 1] = low
    matrix[target, 0] = low
    matrix[target, 1] = high
    return matrix
