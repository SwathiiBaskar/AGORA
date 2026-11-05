"""The evidence update against the channel model it claims to implement (ADR-004).

* T1a: closed-form information content of each channel equals its value computed from
  the channel's probability table.
* T1b: the update's log-odds equal the channel's log-likelihood ratios, observation by
  observation, and drift at the rate the channel's information content predicts.
* T1c: an agent that updates with weight 1 from the true prior is calibrated.
* T1d: the same checks flag agents that are over- or under-confident, so a pass means
  something.
* T1e: against exact rational Bayes, with and without the floor binding.
* T1f: weight ``w`` at strength ``B`` equals weight 1 at strength ``w * B`` equals the
  channel table at strength ``w * B``.
* T1g: no event moves any pair of log-odds by more than its weighted strength.
"""

import math
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from civbox.sim.beliefs import (
    FloatArray,
    floor_beliefs,
    log_normalize,
    point_to_channel,
    rules_out_channel,
)
from civbox.sim.evidence import EvidenceBatch, accumulate_evidence
from civbox.sim.rng import RngStreams
from civbox.sim.update import update_beliefs

from .belief_support import MAX_WEIGHT, random_world

PROPERTY = settings(max_examples=100, deadline=None, derandomize=True, database=None)

OutcomeEvent = tuple[int, float, float]  # (target hypothesis, sign, strength)


def kl_divergence(p: FloatArray, q: FloatArray) -> float:
    return sum(float(a) * math.log(float(a) / float(b)) for a, b in zip(p, q, strict=True) if a > 0)


def point_to_events(strength: float, k: int) -> list[OutcomeEvent]:
    """One event per outcome of the K-ary channel: outcome y points to hypothesis y."""
    return [(y, 1.0, strength) for y in range(k)]


def rules_out_events(target: int, strength: float) -> list[OutcomeEvent]:
    """Outcome 0 rules the target out (sign -1); outcome 1 does not (sign +1)."""
    return [(target, -1.0, strength), (target, 1.0, strength)]


def batch_for(events: list[OutcomeEvent]) -> EvidenceBatch:
    n = len(events)
    return EvidenceBatch.from_components(
        event_ids=np.arange(n, dtype=np.int64),
        component_indices=np.zeros(n, dtype=np.int64),
        targets=np.array([e[0] for e in events], dtype=np.int64),
        signs=np.array([e[1] for e in events]),
        strengths=np.array([e[2] for e in events]),
    )


def observe(
    log_p: FloatArray,
    events: list[OutcomeEvent],
    outcomes: np.ndarray,
    weight: float,
) -> FloatArray:
    """Each agent observes the outcome ``outcomes[i]`` and updates with ``weight``."""
    batch = batch_for(events)
    weights = np.zeros((len(events), log_p.shape[0]))
    weights[outcomes, np.arange(log_p.shape[0])] = weight
    return log_normalize(accumulate_evidence(log_p, batch, weights))


def draw_outcomes(rng: np.random.Generator, matrix: FloatArray, truth: np.ndarray) -> np.ndarray:
    cdf: FloatArray = np.cumsum(matrix, axis=1)
    u: FloatArray = rng.random(int(truth.shape[0]))
    drawn = (u[:, None] > cdf[truth]).sum(axis=1)
    result: np.ndarray = np.minimum(drawn, matrix.shape[1] - 1)
    return result


# --- T1a: information content in closed form ---


@pytest.mark.parametrize("k", [2, 3, 5, 12])
@pytest.mark.parametrize("strength", [0.0, 0.1, 1.0, 5.0, 30.0])
def test_t1a_the_point_to_information_content_has_the_closed_form(k: int, strength: float) -> None:
    matrix = point_to_channel(strength, k)
    closed = strength * (math.exp(strength) - 1.0) / (math.exp(strength) + k - 1.0)
    for j in range(k):
        for other in range(k):
            if j != other:
                assert kl_divergence(matrix[j], matrix[other]) == pytest.approx(
                    closed, rel=1e-9, abs=1e-15
                )
    if k == 2:
        assert closed == pytest.approx(strength * math.tanh(strength / 2.0), rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("strength", [0.0, 0.1, 1.0, 5.0, 30.0])
def test_t1a_the_rule_out_channel_distinguishes_only_the_target_from_the_rest(
    strength: float,
) -> None:
    matrix = rules_out_channel(0, strength, 4)
    closed = strength * math.tanh(strength / 2.0)
    for other in (1, 2, 3):
        assert kl_divergence(matrix[0], matrix[other]) == pytest.approx(closed, rel=1e-9, abs=1e-15)
        assert kl_divergence(matrix[other], matrix[0]) == pytest.approx(closed, rel=1e-9, abs=1e-15)
    for a, b in ((1, 2), (2, 3), (3, 1)):
        assert kl_divergence(matrix[a], matrix[b]) == 0.0


# --- T1b: the update reproduces the channel's likelihood ratios ---


@pytest.mark.parametrize("strength", [0.3, 1.0, 4.0])
def test_t1b_every_outcome_moves_every_pair_of_log_odds_by_the_channel_log_likelihood_ratio(
    strength: float,
) -> None:
    k = 4
    cases = [
        (point_to_channel(strength, k), point_to_events(strength, k)),
        (rules_out_channel(2, strength, k), rules_out_events(2, strength)),
    ]
    for matrix, events in cases:
        start = np.log(np.full((1, k), 1.0 / k))
        for outcome in range(matrix.shape[1]):
            after = observe(start, events, np.array([outcome]), 1.0)
            for j in range(k):
                for other in range(k):
                    expected = math.log(matrix[j, outcome] / matrix[other, outcome])
                    moved = float((after[0, j] - after[0, other]) - (start[0, j] - start[0, other]))
                    assert moved == pytest.approx(expected, abs=1e-12)


def drift_statistics(
    matrix: FloatArray,
    events: list[OutcomeEvent],
    true: int,
    other: int,
    *,
    steps: int,
    agents: int,
    seed: int,
) -> tuple[float, float, float]:
    """Mean per-observation log-odds drift, its predicted value, and its standard error."""
    k = matrix.shape[0]
    rng = RngStreams(seed).generator("drift")
    log_p = np.log(np.full((agents, k), 1.0 / k))
    truth = np.full(agents, true)
    for _ in range(steps):
        log_p = observe(log_p, events, draw_outcomes(rng, matrix, truth), 1.0)
    drift = (log_p[:, true] - log_p[:, other]) / steps

    llr = [
        math.log(matrix[true, y] / matrix[other, y]) if matrix[true, y] > 0 else 0.0
        for y in range(matrix.shape[1])
    ]
    expected = sum(float(matrix[true, y]) * llr[y] for y in range(matrix.shape[1]))
    variance = sum(
        float(matrix[true, y]) * (llr[y] - expected) ** 2 for y in range(matrix.shape[1])
    )
    return float(drift.mean()), expected, math.sqrt(variance / (agents * steps))


@pytest.mark.parametrize("seed", [11, 12])
def test_t1b_log_odds_drift_at_the_rate_given_by_the_point_to_information_content(
    seed: int,
) -> None:
    k, strength = 3, 0.8
    mean, expected, error = drift_statistics(
        point_to_channel(strength, k),
        point_to_events(strength, k),
        0,
        1,
        steps=40,
        agents=4000,
        seed=seed,
    )
    closed = strength * (math.exp(strength) - 1.0) / (math.exp(strength) + k - 1.0)
    assert expected == pytest.approx(closed, rel=1e-12)
    assert abs(mean - expected) <= 5.0 * error


def test_t1b_log_odds_drift_at_the_rate_given_by_the_rule_out_information_content() -> None:
    strength = 1.0
    mean, expected, error = drift_statistics(
        rules_out_channel(0, strength, 3),
        rules_out_events(0, strength),
        0,
        1,
        steps=40,
        agents=4000,
        seed=13,
    )
    assert expected == pytest.approx(strength * math.tanh(strength / 2.0), rel=1e-12)
    assert abs(mean - expected) <= 5.0 * error


def test_t1b_an_indistinguishable_pair_never_drifts() -> None:
    # Ruling out hypothesis 0 says nothing about hypotheses 1 versus 2.
    strength, k = 1.0, 3
    matrix, events = rules_out_channel(0, strength, k), rules_out_events(0, strength)
    rng = RngStreams(14).generator("drift")
    log_p = np.log(np.full((2000, k), 1.0 / k))
    truth = np.full(2000, 1)
    for _ in range(30):
        log_p = observe(log_p, events, draw_outcomes(rng, matrix, truth), 1.0)
    assert np.abs(log_p[:, 1] - log_p[:, 2]).max() <= 1e-12


# --- T1c, T1d: calibration, and the ability to detect its absence ---


PRIOR = np.array([0.5, 0.3, 0.2])
STRENGTH = 0.8
STEPS = 6
AGENTS = 20_000


def calibrated_world(weight: float, seed: int = 100) -> tuple[np.ndarray, FloatArray]:
    k = len(PRIOR)
    rng = RngStreams(seed).generator("world")
    truth = rng.choice(k, size=AGENTS, p=PRIOR)
    matrix, events = point_to_channel(STRENGTH, k), point_to_events(STRENGTH, k)
    log_p = np.tile(np.log(PRIOR), (AGENTS, 1))
    for _ in range(STEPS):
        log_p = observe(log_p, events, draw_outcomes(rng, matrix, truth), weight)
    return truth, np.exp(log_p)


def reliability_violations(
    truth: np.ndarray,
    posterior: FloatArray,
    *,
    bins: int = 10,
    min_count: int = 200,
    sigmas: float = 5.0,
) -> list[tuple[int, int, float]]:
    """Bins in which the events' frequency differs from the stated probability by > sigmas.

    Within a bin, the number of agents whose hypothesis is true has mean ``sum p`` and
    variance ``sum p (1 - p)`` if the stated probabilities are calibrated.
    """
    found = []
    for hypothesis in range(posterior.shape[1]):
        p = posterior[:, hypothesis]
        which = np.minimum((p * bins).astype(int), bins - 1)
        for b in range(bins):
            members = which == b
            if members.sum() < min_count:
                continue
            expected = float(p[members].sum())
            spread = math.sqrt(float((p[members] * (1.0 - p[members])).sum()))
            z = (float((truth[members] == hypothesis).sum()) - expected) / spread
            if abs(z) > sigmas:
                found.append((hypothesis, b, z))
    return found


def test_t1c_the_reference_agent_is_calibrated() -> None:
    truth, posterior = calibrated_world(1.0)
    assert reliability_violations(truth, posterior) == []


def test_t1c_the_reference_agents_expected_posterior_is_the_prior() -> None:
    # Bayesian posteriors are a martingale: their expectation never moves from the prior.
    _, posterior = calibrated_world(1.0)
    for k, prior in enumerate(PRIOR):
        standard_error = posterior[:, k].std() / math.sqrt(AGENTS)
        assert abs(posterior[:, k].mean() - prior) <= 5.0 * standard_error


def test_t1c_the_reference_agent_learns_the_truth() -> None:
    truth, posterior = calibrated_world(1.0)
    assert (posterior.argmax(axis=1) == truth).mean() > 0.6
    assert posterior[np.arange(AGENTS), truth].mean() > PRIOR[truth].mean()


@pytest.mark.parametrize(("weight", "label"), [(2.0, "over-confident"), (0.5, "under-confident")])
def test_t1d_the_calibration_check_flags_miscalibrated_agents(weight: float, label: str) -> None:
    truth, posterior = calibrated_world(weight)
    violations = reliability_violations(truth, posterior)
    assert len(violations) >= 5, f"{label} agent was not flagged"
    assert max(abs(z) for _, _, z in violations) > 10.0


# --- T1e: exact rational arithmetic ---


@PROPERTY
@given(st.integers(0, 2**32 - 1), st.integers(2, 6), st.integers(1, 6))
def test_t1e_the_update_equals_exact_rational_bayes_apart_from_the_floor(
    seed: int, k: int, n_components: int
) -> None:
    rng = np.random.default_rng(seed)
    floor = 1e-9
    counts = rng.integers(1, 50, size=k)
    prior = [Fraction(int(c), int(counts.sum())) for c in counts]

    ratios = [
        Fraction(int(d + rng.integers(1, 5)), int(d)) for d in rng.integers(1, 4, size=n_components)
    ]
    targets = rng.integers(0, k, size=n_components)
    signs = rng.choice([-1, 1], size=n_components)
    weights = rng.integers(0, 4, size=n_components)

    # Exact Bayes: a component with likelihood ratio R and weight w multiplies the
    # likelihood of its target by R ** (sign * w) relative to every other hypothesis.
    exact = list(prior)
    for ratio, target, sign, weight in zip(ratios, targets, signs, weights, strict=True):
        exact[int(target)] *= ratio ** (int(sign) * int(weight))
    total = sum(exact)
    expected = np.array([float(x / total) for x in exact])

    batch = EvidenceBatch.from_components(
        event_ids=np.arange(n_components, dtype=np.int64),
        component_indices=np.zeros(n_components, dtype=np.int64),
        targets=targets.astype(np.int64),
        signs=signs.astype(np.float64),
        strengths=np.array([math.log(r.numerator / r.denominator) for r in ratios]),
    )
    result = update_beliefs(
        np.array([[float(p) for p in prior]]),
        floor=floor,
        batch=batch,
        weights=weights.astype(np.float64)[:, None],
    )[0]

    binds = expected.min() < floor * (1.0 + 1e-6)
    allowance = (k - 1) * floor if binds else 0.0
    assert np.all(np.abs(result - expected) <= 1e-11 * expected + allowance + 1e-300)


# --- T1f: weight is strength ---


@PROPERTY
@given(st.integers(0, 2**32 - 1), st.integers(2, 8), st.floats(0.01, 6.0), st.floats(0.0, 3.0))
def test_t1f_weight_w_at_strength_b_is_strength_w_b_is_the_channel_table(
    seed: int, k: int, strength: float, weight: float
) -> None:
    rng = np.random.default_rng(seed)
    prior = rng.dirichlet(np.ones(k))
    target = int(rng.integers(0, k))
    log_prior = np.log(prior)[None, :]
    unit = np.ones((1, 1))

    def run(sign: float, s: float, w: float) -> FloatArray:
        batch = batch_for([(target, sign, s)])
        posterior: FloatArray = np.exp(
            log_normalize(accumulate_evidence(log_prior, batch, unit * w))
        )[0]
        return posterior

    effective = weight * strength
    point = point_to_channel(effective, k)[:, target] * prior
    rules = rules_out_channel(target, effective, k)
    for sign, table_column in (
        (1.0, point),
        (-1.0, rules[:, 0] * prior),
        (1.0, rules[:, 1] * prior),
    ):
        reweighted = run(sign, strength, weight)
        restrengthened = run(sign, effective, 1.0)
        assert np.allclose(reweighted, restrengthened, rtol=1e-12, atol=0.0)
        assert np.allclose(reweighted, table_column / table_column.sum(), rtol=1e-10, atol=0.0)


# --- T1g: a bound on how far one event can move the log-odds ---


@PROPERTY
@given(st.integers(0, 2**32 - 1), st.integers(1, 9), st.integers(2, 20))
def test_t1g_no_pair_of_log_odds_moves_more_than_the_weighted_strength_of_the_events(
    seed: int, n: int, k: int
) -> None:
    world = random_world(seed, n_agents=n, n_hypotheses=k, floor=1e-12 / k, max_strength=12.0)
    batch = world.batch()
    weights = world.weights_for(batch)
    before = np.log(world.beliefs)
    after = accumulate_evidence(before, batch, weights)
    shift = after - before

    budget = np.zeros(n)
    for c in range(batch.n_components):
        budget += weights[batch.component_event[c]] * batch.strength[c]
    spread = shift.max(axis=1) - shift.min(axis=1)
    assert (spread <= budget + 1e-9).all()
    assert (spread <= MAX_WEIGHT * batch.strength.sum() + 1e-9).all()


@PROPERTY
@given(st.integers(0, 2**32 - 1), st.integers(2, 20), st.floats(0.0, 50.0), st.floats(0.0, 3.0))
def test_t1g_the_bound_is_attained_by_a_single_component(
    seed: int, k: int, strength: float, weight: float
) -> None:
    target = seed % k
    before = np.log(np.random.default_rng(seed).dirichlet(np.ones(k)))[None, :]
    batch = batch_for([(target, 1.0, strength)])
    after = accumulate_evidence(before, batch, np.full((1, 1), weight))
    others = [h for h in range(k) if h != target]
    moved = (after[0, target] - after[0, others]) - (before[0, target] - before[0, others])
    assert np.allclose(moved, weight * strength, rtol=1e-12, atol=1e-12)


def test_floor_beliefs_is_the_only_stage_that_departs_from_exact_bayes() -> None:
    # With a floor too small to bind, the full update is exact Bayes to rounding.
    prior = np.array([[0.2, 0.3, 0.5]])
    batch = batch_for([(0, 1.0, 1.5)])
    result = update_beliefs(prior, floor=1e-12, batch=batch, weights=np.ones((1, 1)))
    exact = prior[0] * np.array([math.exp(1.5), 1.0, 1.0])
    assert np.allclose(result[0], exact / exact.sum(), rtol=1e-14, atol=0.0)
    assert floor_beliefs(result, 1e-12) is not result
