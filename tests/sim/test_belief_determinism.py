"""Numerical determinism of the belief update (ADR-003, ADR-004).

Results must not depend on which process computes them, how many threads the numerical
libraries are allowed to use, or the order of additions. This does not claim identical
results across different CPU dispatch states or NumPy builds; see ADR-003.
"""

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from civbox.sim.evidence import EvidenceBatch, accumulate_evidence
from civbox.sim.pooling import InfluenceEdges, accumulate_by_receiver
from civbox.sim.rng import RngStreams

from .belief_support import reference_update_digest

REPOSITORY = Path(__file__).resolve().parents[2]
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
UNIT_ROUNDOFF = 2.0**-53


def digest_in_fresh_process(seed: int, threads: int | None = None) -> str:
    environment = dict(os.environ)
    if threads is not None:
        environment.update({name: str(threads) for name in THREAD_VARIABLES})
    code = (
        "from tests.sim.belief_support import reference_update_digest; "
        f"print(reference_update_digest({seed}))"
    )
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


def test_s8_the_same_update_is_bit_identical_across_processes() -> None:
    in_process = reference_update_digest(123)
    assert reference_update_digest(123) == in_process
    assert digest_in_fresh_process(123) == in_process


def test_s8_different_inputs_give_different_digests() -> None:
    assert reference_update_digest(1) != reference_update_digest(2)


@pytest.mark.parametrize("threads", [1, 8])
def test_s4_the_update_does_not_depend_on_the_thread_count_of_numerical_libraries(
    threads: int,
) -> None:
    assert digest_in_fresh_process(123, threads) == reference_update_digest(123)


def forward_error_bound(terms: list[float]) -> float:
    """Recursive summation of n numbers errs by at most gamma_(n-1) * sum(|x|)."""
    n = len(terms)
    gamma = (n - 1) * UNIT_ROUNDOFF / (1.0 - (n - 1) * UNIT_ROUNDOFF)
    return gamma * sum(abs(x) for x in terms)


def test_s5_evidence_accumulation_stays_within_the_forward_error_bound() -> None:
    rng = RngStreams(51).generator("s5")
    n_agents, n_components = 8, 300
    batch = EvidenceBatch.from_components(
        event_ids=np.arange(n_components, dtype=np.int64),
        component_indices=np.zeros(n_components, dtype=np.int64),
        targets=np.zeros(n_components, dtype=np.int64),
        signs=rng.choice(np.array([-1.0, 1.0]), size=n_components),
        strengths=rng.random(n_components) * 10.0 ** rng.integers(-3, 7, size=n_components),
    )
    weights = rng.random((n_components, n_agents)) * 3.0
    start = rng.normal(size=(n_agents, 2)) * 1e3
    result = accumulate_evidence(start, batch, weights)

    signed = batch.signed_strength
    rounding_seen = False
    for agent in range(n_agents):
        terms = [float(start[agent, 0])] + [
            float(weights[c, agent]) * float(signed[c]) for c in range(n_components)
        ]
        exact = math.fsum(terms)
        error = abs(float(result[agent, 0]) - exact)
        assert error <= forward_error_bound(terms)
        rounding_seen |= error > 0.0
    # Without rounding the bound would hold vacuously.
    assert rounding_seen


def test_s5_accumulation_by_receiver_stays_within_the_forward_error_bound() -> None:
    rng = RngStreams(52).generator("s5")
    n_edges = 500
    edges = InfluenceEdges.from_arrays(
        n_edges + 1,
        np.zeros(n_edges, dtype=np.int64),
        np.arange(1, n_edges + 1, dtype=np.int64),
        np.ones(n_edges),
    )
    values = rng.normal(size=(n_edges, 3)) * 10.0 ** rng.integers(-4, 8, size=(n_edges, 3))
    totals = accumulate_by_receiver(edges, values)

    rounding_seen = False
    for k in range(3):
        terms = [float(v) for v in values[:, k]]
        exact = math.fsum(terms)
        error = abs(float(totals[0, k]) - exact)
        assert error <= forward_error_bound(terms)
        rounding_seen |= error > 0.0
    assert rounding_seen
