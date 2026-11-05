import math
from typing import Any

import numpy as np
import pytest

from civbox.sim.beliefs import (
    DEFAULT_BELIEF_FLOOR,
    check_floor,
    floor_beliefs,
    hellinger_distance,
    log_normalize,
    normalized_entropy,
    point_to_channel,
    rules_out_channel,
    sum_over_hypotheses,
    validate_beliefs,
)
from civbox.sim.rng import RngStreams


def rows(*values: list[float]) -> Any:
    return np.array(values, dtype=np.float64)


# --- the floor ---------------------------------------------------------------


def test_the_default_floor_is_a_valid_floor_for_many_hypotheses() -> None:
    check_floor(1000, DEFAULT_BELIEF_FLOOR)


@pytest.mark.parametrize(
    ("k", "floor"), [(1, 0.1), (2, 0.0), (2, -1e-9), (2, 0.5), (3, 1 / 3), (3, 0.4)]
)
def test_invalid_floors_are_rejected(k: int, floor: float) -> None:
    with pytest.raises(ValueError, match=r"hypotheses are required|floor must satisfy"):
        check_floor(k, floor)


def test_rows_already_above_the_floor_are_returned_bit_for_bit() -> None:
    q = rows([0.7, 0.2, 0.1], [1 / 3, 1 / 3, 1 / 3], [0.5, 0.25, 0.25])
    assert np.array_equal(floor_beliefs(q, 1e-3), q)


def test_a_binding_floor_brings_the_smallest_entry_exactly_to_the_floor() -> None:
    p = floor_beliefs(rows([0.9995, 0.0005]), 1e-3)
    assert p[0, 1] == 1e-3
    assert math.isclose(p[0, 0], 0.999, rel_tol=1e-15)
    assert abs(float(p.sum()) - 1.0) <= 1e-15


def test_zero_entries_are_raised_to_the_floor() -> None:
    p = floor_beliefs(rows([1.0, 0.0, 0.0]), 1e-6)
    assert p.min() == 1e-6
    assert abs(float(p.sum()) - 1.0) <= 1e-15
    assert p[0, 0] > p[0, 1] == p[0, 2]


def test_the_floor_is_idempotent_bit_for_bit() -> None:
    rng = RngStreams(3).generator("floor")
    for k, floor in ((2, 1e-3), (3, 1e-9), (7, 0.01), (50, 1e-4)):
        q = rng.dirichlet(np.full(k, 0.05), size=300)
        once = floor_beliefs(q, floor)
        assert np.array_equal(floor_beliefs(once, floor), once)


def test_an_unconditional_mix_with_the_uniform_distribution_would_not_be_idempotent() -> None:
    # Documents why the floor mixes only as much as needed: mixing every row every time
    # drags confident beliefs toward uniform without any evidence.
    eps, k = 1e-3, 2

    def mix(x: Any) -> Any:
        return (1 - k * eps) * x + eps

    once = mix(rows([0.9995, 0.0005]))
    twice = mix(once)
    assert not np.allclose(once, twice, atol=1e-4)
    # The floor used here leaves an already floored row exactly where it is.
    floored = floor_beliefs(rows([0.9995, 0.0005]), eps)
    assert np.array_equal(floor_beliefs(floored, eps), floored)


def test_the_floor_preserves_the_order_of_entries() -> None:
    rng = RngStreams(4).generator("floor")
    q = rng.dirichlet(np.full(6, 0.05), size=500)
    p = floor_beliefs(q, 1e-3)
    for a in range(6):
        for b in range(6):
            assert not ((q[:, a] > q[:, b]) & (p[:, a] < p[:, b])).any()


def test_the_floor_error_bound_is_k_minus_one_times_the_floor_and_is_attained() -> None:
    rng = RngStreams(5).generator("floor")
    slack = 4 * np.finfo(np.float64).eps  # the compared values are close to 1
    for k in (2, 3, 7, 50):
        for eps in (1e-9, 1e-6, 1e-3, 0.4 / k):
            bound = (k - 1) * eps
            q = rng.dirichlet(np.full(k, 0.02), size=300)
            certain = np.zeros((1, k))
            certain[0, 0] = 1.0
            q = np.vstack([q, certain])
            error = np.abs(floor_beliefs(q, eps) - q).max(axis=1)
            assert (error <= bound * (1 + 1e-9) + slack).all()
            assert abs(float(error[-1]) - bound) <= slack


def test_the_floor_moves_every_pair_of_entries_toward_each_other() -> None:
    rng = RngStreams(6).generator("floor")
    q = rng.dirichlet(np.full(5, 0.03), size=400)
    p = floor_beliefs(q, 1e-3)
    with np.errstate(divide="ignore", invalid="ignore"):
        before = np.log(q[:, :, None] / q[:, None, :])
    after = np.log(p[:, :, None] / p[:, None, :])
    finite = np.isfinite(before)
    lower = np.minimum(before, 0.0)[finite] - 1e-12
    upper = np.maximum(before, 0.0)[finite] + 1e-12
    assert (after[finite] >= lower).all()
    assert (after[finite] <= upper).all()


def test_the_floor_does_not_modify_its_input() -> None:
    q = rows([1.0, 0.0])
    snapshot = q.copy()
    floor_beliefs(q, 1e-3)
    assert np.array_equal(q, snapshot)


@pytest.mark.parametrize("bad", [np.zeros((2, 3), dtype=np.float32), np.zeros(3), [[0.5, 0.5]]])
def test_the_floor_rejects_wrong_array_types(bad: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        floor_beliefs(bad, 1e-3)


# --- validation of supplied beliefs -------------------------------------------


def test_beliefs_on_the_floored_simplex_are_accepted() -> None:
    validate_beliefs(rows([0.5, 0.5], [0.999, 0.001]), 1e-3)


@pytest.mark.parametrize(
    ("beliefs", "message"),
    [
        (rows([1.0, 0.0]), "at least the floor"),
        (rows([0.5, 0.4]), "sum to 1"),
        (rows([0.5, np.nan]), "finite"),
        (rows([0.5, np.inf]), "finite"),
    ],
)
def test_invalid_beliefs_are_rejected(beliefs: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_beliefs(beliefs, 1e-3)


# --- reductions ---------------------------------------------------------------


def test_the_hypothesis_sum_is_a_strict_left_to_right_sum() -> None:
    rng = RngStreams(7).generator("sum")
    values = rng.normal(size=(40, 23)) * 10.0 ** rng.integers(-8, 8, size=(40, 23))
    expected = np.empty(40)
    for row in range(40):
        total = float(values[row, 0])
        for k in range(1, 23):
            total += float(values[row, k])
        expected[row] = total
    assert np.array_equal(sum_over_hypotheses(values), expected)


def test_log_normalize_produces_rows_whose_exponentials_sum_to_one() -> None:
    rng = RngStreams(8).generator("norm")
    z = rng.normal(size=(200, 9)) * 30
    assert np.abs(np.exp(log_normalize(z)).sum(axis=1) - 1.0).max() < 1e-14


def test_log_normalize_ignores_a_constant_added_to_a_row() -> None:
    rng = RngStreams(9).generator("norm")
    z = rng.normal(size=(50, 5))
    shifted = z + rng.normal(size=(50, 1)) * 100
    assert np.allclose(log_normalize(z), log_normalize(shifted), atol=1e-11)


def test_log_normalize_survives_huge_magnitudes() -> None:
    z = rows([1e5, 1e5 - 3.0], [-1e5, -1e5 + 2.0], [800.0, -800.0])
    result = log_normalize(z)
    assert np.isfinite(result).all()
    assert np.abs(np.exp(result).sum(axis=1) - 1.0).max() < 1e-14


# --- entropy and distance -------------------------------------------------------


def test_normalised_entropy_of_the_extremes() -> None:
    result = normalized_entropy(rows([0.25] * 4, [1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]))
    assert result[0] == pytest.approx(1.0, abs=1e-15)
    assert result[1] == 0.0
    assert result[2] == pytest.approx(0.5, abs=1e-15)


def test_normalised_entropy_stays_in_the_unit_interval() -> None:
    rng = RngStreams(10).generator("entropy")
    for k in (2, 5, 50):
        for alpha in (0.01, 0.5, 5.0):
            p = rng.dirichlet(np.full(k, alpha), size=200)
            h = normalized_entropy(p)
            assert ((h >= 0.0) & (h <= 1.0)).all()


def test_hellinger_distance_known_values() -> None:
    d = hellinger_distance(
        rows([1.0, 0.0], [0.5, 0.5], [0.3, 0.7]), rows([0.0, 1.0], [0.5, 0.5], [0.3, 0.7])
    )
    assert d[0] == 1.0
    assert d[1] == 0.0
    assert d[2] == 0.0
    half = hellinger_distance(rows([1.0, 0.0]), rows([0.5, 0.5]))
    assert half[0] == pytest.approx(math.sqrt(1 - math.sqrt(0.5)), rel=1e-15)


def test_hellinger_distance_is_symmetric_and_bounded() -> None:
    rng = RngStreams(11).generator("hellinger")
    p = rng.dirichlet(np.full(6, 0.1), size=300)
    q = rng.dirichlet(np.full(6, 0.1), size=300)
    d = hellinger_distance(p, q)
    assert np.array_equal(d, hellinger_distance(q, p))
    assert ((d >= 0.0) & (d <= 1.0)).all()


# --- channels ---------------------------------------------------------------------


@pytest.mark.parametrize("k", [2, 3, 10])
@pytest.mark.parametrize("strength", [0.0, 0.5, 3.0, 40.0])
def test_point_to_channels_are_row_stochastic_with_the_stated_likelihood_ratio(
    k: int, strength: float
) -> None:
    matrix = point_to_channel(strength, k)
    assert matrix.shape == (k, k)
    assert np.abs(matrix.sum(axis=1) - 1.0).max() < 1e-15
    assert (matrix >= 0.0).all()
    hit, miss = matrix[0, 0], matrix[0, 1]
    assert math.log(hit / miss) == pytest.approx(strength, abs=1e-12)
    assert np.allclose(np.diag(matrix), hit, rtol=0, atol=0)


@pytest.mark.parametrize("strength", [0.0, 0.5, 3.0, 40.0])
def test_rules_out_channels_are_row_stochastic_with_the_stated_likelihood_ratio(
    strength: float,
) -> None:
    matrix = rules_out_channel(1, strength, 4)
    assert matrix.shape == (4, 2)
    assert np.abs(matrix.sum(axis=1) - 1.0).max() < 1e-15
    # Seeing "rules out" is e^-B times as likely under the target as under any other.
    assert math.log(matrix[1, 0] / matrix[0, 0]) == pytest.approx(-strength, abs=1e-12)
    # Seeing "does not rule out" is e^B times as likely under the target.
    assert math.log(matrix[1, 1] / matrix[0, 1]) == pytest.approx(strength, abs=1e-12)
    assert np.array_equal(matrix[[0, 2, 3]], np.tile(matrix[0], (3, 1)))


def test_a_zero_strength_channel_is_uninformative() -> None:
    assert np.allclose(point_to_channel(0.0, 5), 0.2, rtol=0, atol=1e-16)
    assert np.allclose(rules_out_channel(2, 0.0, 5), 0.5, rtol=0, atol=1e-16)


def test_extreme_strengths_do_not_overflow_or_produce_nan() -> None:
    for matrix in (point_to_channel(900.0, 4), rules_out_channel(0, 900.0, 4)):
        assert np.isfinite(matrix).all()
        assert np.abs(matrix.sum(axis=1) - 1.0).max() < 1e-15


@pytest.mark.parametrize("strength", [-0.1, math.nan, math.inf])
def test_invalid_strengths_are_rejected(strength: float) -> None:
    with pytest.raises(ValueError, match="strength"):
        point_to_channel(strength, 3)
    with pytest.raises(ValueError, match="strength"):
        rules_out_channel(0, strength, 3)


def test_invalid_channel_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least 2 hypotheses"):
        point_to_channel(1.0, 1)
    with pytest.raises(ValueError, match="target"):
        rules_out_channel(3, 1.0, 3)
    with pytest.raises(ValueError, match="target"):
        rules_out_channel(-1, 1.0, 3)
