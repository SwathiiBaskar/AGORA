from typing import Any

import numpy as np
import pytest

from civbox.sim import exposure
from civbox.sim.exposure import (
    MAX_COUNT,
    MAX_TICK,
    ExposureBuffer,
    ExposureLedger,
)


def ints(*v: int) -> Any:
    return np.array(v, dtype=np.int64)


def ledger(n_agents: int = 4, n_claims: int = 3) -> ExposureLedger:
    return ExposureLedger(n_agents, n_claims)


def frozen(ledger_: ExposureLedger) -> tuple[Any, ...]:
    return (
        ledger_.available.copy(),
        ledger_.counts.copy(),
        ledger_.first_tick.copy(),
        ledger_.last_tick.copy(),
        ledger_.committed_tick,
    )


def same(before: tuple[Any, ...], after: tuple[Any, ...]) -> bool:
    return all(np.array_equal(a, b) for a, b in zip(before, after, strict=True))


# --- availability is not exposure ---


def test_a_new_ledger_has_nothing_available_and_nothing_observed() -> None:
    fresh = ledger()
    assert not fresh.available.any()
    assert not fresh.counts.any()
    assert (fresh.first_tick == -1).all()
    assert (fresh.last_tick == -1).all()
    assert fresh.committed_tick == -1


def test_making_a_claim_available_does_not_expose_the_agent_to_it() -> None:
    fresh = ledger()
    fresh.make_available(ints(1), ints(2))
    assert fresh.is_available(1, 2)
    assert not fresh.was_exposed(1, 2)
    assert fresh.counts[1, 2] == 0


def test_exposure_is_per_agent_and_per_claim() -> None:
    fresh = ledger()
    fresh.make_available(ints(0, 0, 1), ints(0, 1, 0))
    fresh.commit(ints(0), ints(0), tick=3)
    assert fresh.was_exposed(0, 0)
    assert not fresh.was_exposed(0, 1)  # same agent, another available claim
    assert not fresh.was_exposed(1, 0)  # same claim, another agent who could see it
    assert fresh.is_available(0, 1)
    assert fresh.is_available(1, 0)


def test_availability_is_idempotent_and_never_withdrawn_by_exposure() -> None:
    fresh = ledger()
    fresh.make_available(ints(0, 0), ints(1, 1))
    fresh.make_available(ints(0), ints(1))
    fresh.commit(ints(0), ints(1), tick=0)
    assert fresh.is_available(0, 1)
    assert int(fresh.available.sum()) == 1


# --- observing requires availability ---


def test_an_observation_of_an_unavailable_claim_is_rejected_naming_the_pair() -> None:
    fresh = ledger()
    with pytest.raises(ValueError, match="agent 2 cannot observe claim 1"):
        fresh.commit(ints(2), ints(1), tick=0)


def test_a_rejected_commit_changes_nothing_even_when_part_of_it_was_valid() -> None:
    fresh = ledger()
    fresh.make_available(ints(0), ints(0))
    before = frozen(fresh)
    with pytest.raises(ValueError, match="never made available"):
        fresh.commit(ints(0, 0, 3), ints(0, 0, 2), tick=7)  # the first two are valid
    assert same(before, frozen(fresh))


def test_check_commit_validates_without_changing_anything() -> None:
    fresh = ledger()
    fresh.make_available(ints(0), ints(0))
    before = frozen(fresh)
    fresh.check_commit(ints(0), ints(0), tick=5)
    assert same(before, frozen(fresh))
    with pytest.raises(ValueError, match="never made available"):
        fresh.check_commit(ints(1), ints(1), tick=5)


# --- what a commit records ---


def test_a_pair_observed_several_times_in_one_commit_counts_each_time() -> None:
    fresh = ledger()
    fresh.make_available(ints(1), ints(2))
    fresh.commit(ints(1, 1, 1), ints(2, 2, 2), tick=4)
    assert fresh.counts[1, 2] == 3


def test_first_and_last_exposure_ticks_are_tracked_separately() -> None:
    fresh = ledger()
    fresh.make_available(ints(0), ints(0))
    fresh.commit(ints(0), ints(0), tick=2)
    fresh.commit(ints(0), ints(0), tick=9)
    assert fresh.first_tick[0, 0] == 2
    assert fresh.last_tick[0, 0] == 9
    assert fresh.counts[0, 0] == 2
    assert fresh.first_tick[1, 1] == -1


def test_commits_in_the_same_tick_accumulate() -> None:
    fresh = ledger()
    fresh.make_available(ints(0), ints(0))
    fresh.commit(ints(0), ints(0), tick=5)
    fresh.commit(ints(0), ints(0), tick=5)
    assert fresh.counts[0, 0] == 2
    assert fresh.committed_tick == 5


def test_an_empty_commit_still_advances_the_committed_tick() -> None:
    fresh = ledger()
    fresh.commit(ints(), ints(), tick=6)
    assert fresh.committed_tick == 6
    with pytest.raises(ValueError, match="earlier than"):
        fresh.commit(ints(), ints(), tick=5)


def test_time_cannot_run_backwards() -> None:
    fresh = ledger()
    fresh.make_available(ints(0), ints(0))
    fresh.commit(ints(0), ints(0), tick=8)
    before = frozen(fresh)
    with pytest.raises(ValueError, match="earlier than"):
        fresh.commit(ints(0), ints(0), tick=7)
    assert same(before, frozen(fresh))


@pytest.mark.parametrize("tick", [-1, MAX_TICK + 1, True, 1.5, "3", None])
def test_invalid_ticks_are_rejected(tick: Any) -> None:
    with pytest.raises(ValueError, match="tick must be an integer"):
        ledger().commit(ints(), ints(), tick=tick)


def test_the_largest_tick_is_accepted() -> None:
    fresh = ledger()
    fresh.make_available(ints(0), ints(0))
    fresh.commit(ints(0), ints(0), tick=MAX_TICK)
    assert fresh.last_tick[0, 0] == MAX_TICK


def test_a_count_that_would_overflow_is_rejected_and_changes_nothing() -> None:
    fresh = ledger()
    fresh.make_available(ints(0), ints(0))
    fresh._counts[0, 0] = MAX_COUNT
    before = frozen(fresh)
    with pytest.raises(ValueError, match="32-bit"):
        fresh.commit(ints(0), ints(0), tick=1)
    assert same(before, frozen(fresh))


# --- queries ---


def test_counts_for_returns_claims_by_agents_as_floats() -> None:
    fresh = ledger(3, 4)
    fresh.make_available(ints(0, 1, 2), ints(1, 1, 3))
    fresh.commit(ints(0, 1, 1, 2), ints(1, 1, 1, 3), tick=0)
    result = fresh.counts_for(ints(3, 1))
    assert result.dtype == np.float64
    assert result.tolist() == [[0.0, 0.0, 1.0], [1.0, 2.0, 0.0]]


def test_queries_reject_out_of_range_indices() -> None:
    fresh = ledger(2, 2)
    for call in (
        lambda: fresh.was_exposed(2, 0),
        lambda: fresh.was_exposed(0, 2),
        lambda: fresh.is_available(-1, 0),
        lambda: fresh.counts_for(ints(2)),
        lambda: fresh.counts_for(ints(-1)),
    ):
        with pytest.raises(ValueError, match="outside"):
            call()


def test_the_arrays_are_read_only_views() -> None:
    fresh = ledger()
    for array in (fresh.available, fresh.counts, fresh.first_tick, fresh.last_tick):
        with pytest.raises(ValueError, match="read-only"):
            array[0, 0] = 1


def test_a_copy_is_independent_of_the_original() -> None:
    fresh = ledger()
    fresh.make_available(ints(0), ints(0))
    snapshot = fresh.copy()
    fresh.commit(ints(0), ints(0), tick=1)
    assert snapshot.counts[0, 0] == 0
    assert snapshot.committed_tick == -1
    snapshot.make_available(ints(1), ints(1))
    assert not fresh.is_available(1, 1)


# --- construction and limits ---


def test_a_ledger_may_have_no_claims() -> None:
    empty = ledger(3, 0)
    empty.commit(ints(), ints(), tick=0)
    assert empty.counts_for(ints()).shape == (0, 3)


def test_oversized_ledgers_are_refused_before_allocating(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="exceeds the ledger limit"):
        ExposureLedger(100_000, 100_000)
    monkeypatch.setattr(exposure, "MAX_LEDGER_CELLS", 100)
    ExposureLedger(10, 10)
    with pytest.raises(ValueError, match="exceeds the ledger limit"):
        ExposureLedger(10, 11)


@pytest.mark.parametrize(("agents", "claims"), [(0, 3), (-1, 3), (3, -1)])
def test_invalid_ledger_sizes_are_rejected(agents: int, claims: int) -> None:
    with pytest.raises(ValueError, match="at least one agent"):
        ExposureLedger(agents, claims)


@pytest.mark.parametrize(
    ("agents", "claims", "message"),
    [
        (ints(0, 1), ints(0), "same length"),
        (ints(4), ints(0), "agent index"),
        (ints(-1), ints(0), "agent index"),
        (ints(0), ints(3), "claim index"),
        (ints(0), ints(-1), "claim index"),
    ],
)
def test_invalid_pairs_are_rejected_everywhere(agents: Any, claims: Any, message: str) -> None:
    fresh = ledger()
    with pytest.raises(ValueError, match=message):
        fresh.make_available(agents, claims)
    with pytest.raises(ValueError, match=message):
        fresh.commit(agents, claims, tick=0)


def test_pairs_must_be_int64_arrays() -> None:
    with pytest.raises(TypeError, match="int64"):
        ledger().make_available(np.array([0], dtype=np.int32), ints(0))
    with pytest.raises(TypeError, match="numpy array"):
        ledger().make_available([0], ints(0))  # type: ignore[arg-type]


# --- the transient buffer ---


def test_the_buffer_keeps_pairs_in_insertion_order() -> None:
    buffer = ExposureBuffer(5, 5)
    buffer.add(ints(3, 1), ints(0, 2))
    buffer.add(ints(1), ints(1))
    agents, claims = buffer.pairs()
    assert agents.tolist() == [3, 1, 1]
    assert claims.tolist() == [0, 2, 1]
    assert buffer.n_pending == 3


def test_an_empty_buffer_has_no_pairs() -> None:
    agents, claims = ExposureBuffer(2, 2).pairs()
    assert agents.size == 0
    assert claims.size == 0
    assert agents.dtype == claims.dtype == np.int64


def test_the_buffer_copies_what_it_is_given_and_can_be_cleared() -> None:
    buffer = ExposureBuffer(3, 3)
    source = ints(1)
    buffer.add(source, ints(2))
    source[0] = 99
    assert buffer.pairs()[0].tolist() == [1]
    buffer.clear()
    assert buffer.n_pending == 0
    assert buffer.pairs()[0].size == 0


@pytest.mark.parametrize(
    ("agents", "claims", "message"),
    [
        (ints(0, 1), ints(0), "same length"),
        (ints(3), ints(0), "agent"),
        (ints(0), ints(3), "claim"),
    ],
)
def test_the_buffer_validates_what_it_is_given(agents: Any, claims: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ExposureBuffer(3, 3).add(agents, claims)


def test_pending_observations_are_invisible_to_the_ledger_until_committed() -> None:
    fresh = ledger()
    fresh.make_available(ints(0), ints(0))
    buffer = ExposureBuffer(4, 3)
    buffer.add(ints(0, 0), ints(0, 0))
    assert not fresh.was_exposed(0, 0)
    assert fresh.counts_for(ints(0))[0, 0] == 0.0
    agents, claims = buffer.pairs()
    fresh.commit(agents, claims, tick=1)
    assert fresh.counts_for(ints(0))[0, 0] == 2.0


# --- consistency checks ---


def corrupted(mutation: Any) -> ExposureLedger:
    fresh = ledger()
    fresh.make_available(ints(0, 1), ints(0, 1))
    fresh.commit(ints(0), ints(0), tick=3)
    mutation(fresh)
    return fresh


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda x: x._available.__setitem__((0, 0), False), "never available"),
        (lambda x: x._counts.__setitem__((1, 1), 1), "never available|disagree"),
        (lambda x: x._first.__setitem__((0, 0), -1), "disagree"),
        (lambda x: x._last.__setitem__((1, 1), 2), "disagree"),
        (lambda x: x._first.__setitem__((0, 0), 9), "out of order"),
        (lambda x: x._last.__setitem__((0, 0), 99), "out of order"),
    ],
)
def test_validate_detects_each_kind_of_corruption(mutation: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        corrupted(mutation).validate()


def test_a_ledger_built_only_through_its_transitions_is_always_valid() -> None:
    fresh = ledger(5, 5)
    rng = np.random.default_rng(0)
    for tick in range(20):
        fresh.make_available(
            rng.integers(0, 5, 4).astype(np.int64), rng.integers(0, 5, 4).astype(np.int64)
        )
        a, c = np.nonzero(fresh.available)
        pick = rng.integers(0, a.size, 6) if a.size else np.array([], dtype=np.int64)
        fresh.commit(a[pick].astype(np.int64), c[pick].astype(np.int64), tick)
        fresh.validate()
