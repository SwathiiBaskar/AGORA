from typing import Any

import numpy as np
import pytest

from civbox.sim.memory import AgentMemory


class WriteRecorder(np.ndarray[Any, np.dtype[np.int32]]):
    """An array that remembers the index it was written through."""

    writes: list[Any]

    def __setitem__(self, key: Any, value: Any) -> None:
        self.writes.append(key)
        super().__setitem__(key, value)


def ints(*v: int) -> Any:
    return np.array(v, dtype=np.int64)


def memory(*capacity: int, n_claims: int = 20) -> AgentMemory:
    return AgentMemory(ints(*capacity), n_claims)


# --- the ring ---


def test_an_agent_remembers_what_it_saw_in_the_order_it_saw_it() -> None:
    m = memory(5)
    m.record(ints(0), ints(7), tick=1)
    m.record(ints(0), ints(3), tick=2)
    assert m.recent(0) == [(7, 1), (3, 2)]
    assert m.size().tolist() == [2]
    assert m.written.tolist() == [2]


def test_a_full_memory_forgets_the_oldest_record_first() -> None:
    m = memory(3)
    for tick, claim in enumerate([10, 11, 12, 13, 14]):
        m.record(ints(0), ints(claim), tick=tick)
    assert m.recent(0) == [(12, 2), (13, 3), (14, 4)]
    assert m.size().tolist() == [3]
    assert m.written.tolist() == [5]


def test_agents_with_larger_capacity_remember_more_of_the_same_history() -> None:
    m = memory(2, 5)
    for tick in range(5):
        m.record(ints(0, 1), ints(tick, tick), tick=tick)
    assert [c for c, _ in m.recent(0)] == [3, 4]
    assert [c for c, _ in m.recent(1)] == [0, 1, 2, 3, 4]


def test_an_agent_with_no_capacity_remembers_nothing() -> None:
    m = memory(0, 2)
    m.record(ints(0, 1), ints(4, 5), tick=0)
    assert m.recent(0) == []
    assert m.recent(1) == [(5, 0)]
    assert m.written.tolist() == [0, 1]
    assert m.remembered_counts(ints(4, 5)).tolist() == [[0.0, 0.0], [0.0, 1.0]]


def test_a_population_where_nobody_remembers_has_empty_arrays() -> None:
    m = memory(0, 0, 0)
    assert m.width == 0
    m.record(ints(0, 1, 2), ints(1, 2, 3), tick=1)
    assert m.claims.shape == (3, 0)
    assert m.remembered_counts(ints(1)).tolist() == [[0.0, 0.0, 0.0]]
    m.validate()


def test_more_records_than_room_in_one_write_keeps_only_the_latest_in_order() -> None:
    m = memory(3)
    m.record(ints(0, 0, 0, 0, 0), ints(1, 2, 3, 4, 5), tick=9)
    assert m.recent(0) == [(3, 9), (4, 9), (5, 9)]
    assert m.written.tolist() == [5]


def test_no_slot_is_written_twice_in_one_write() -> None:
    # The result must not depend on how NumPy orders repeated indices, so records that a
    # later record of the same agent would overwrite are never written at all.
    m = memory(3, 1, 4, 0)
    recorder = m._claims.view(WriteRecorder)
    recorder.writes = []
    m._claims = recorder
    agents = ints(0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 3)
    m.record(agents, ints(*range(14)), tick=1)
    assert recorder.writes, "nothing was written"
    for rows, columns in recorder.writes:
        targets = list(zip(rows.tolist(), columns.tolist(), strict=True))
        assert len(targets) == len(set(targets))
    assert m.recent(0) == [(2, 1), (3, 1), (4, 1)]
    assert m.recent(1) == [(6, 1)]
    assert m.recent(2) == [(9, 1), (10, 1), (11, 1), (12, 1)]
    assert m.recent(3) == []


def test_when_the_same_agent_appears_twice_with_room_for_one_the_later_record_wins() -> None:
    m = memory(1)
    m.record(ints(0, 0), ints(6, 8), tick=0)
    assert m.recent(0) == [(8, 0)]


def test_one_agents_records_keep_their_order_when_other_agents_are_interleaved() -> None:
    m = memory(4, 4, 4)
    m.record(ints(0, 1, 0, 2, 1, 0), ints(10, 15, 11, 18, 16, 12), tick=3)
    assert [c for c, _ in m.recent(0)] == [10, 11, 12]
    assert [c for c, _ in m.recent(1)] == [15, 16]
    assert [c for c, _ in m.recent(2)] == [18]


def test_a_write_continues_where_the_previous_one_stopped() -> None:
    m = memory(4)
    m.record(ints(0, 0, 0), ints(1, 2, 3), tick=1)
    m.record(ints(0, 0, 0), ints(4, 5, 6), tick=2)
    assert m.recent(0) == [(3, 1), (4, 2), (5, 2), (6, 2)]


def test_the_same_claim_seen_repeatedly_is_remembered_repeatedly() -> None:
    m = memory(4)
    m.record(ints(0, 0, 0), ints(5, 5, 9), tick=0)
    assert m.remembered_counts(ints(5, 9, 1)).tolist() == [[2.0], [1.0], [0.0]]


def test_memory_stays_bounded_however_much_is_written() -> None:
    capacity = ints(0, 1, 3, 8)
    m = AgentMemory(capacity, 50)
    shape = m.claims.shape
    rng = np.random.default_rng(0)
    for tick in range(300):
        agents = rng.integers(0, 4, 40).astype(np.int64)
        m.record(agents, rng.integers(0, 50, 40).astype(np.int64), tick)
        assert m.claims.shape == shape
        assert (m.size() <= capacity).all()
        assert ((m.claims != -1).sum(axis=1) == m.size()).all()
    m.validate()


def test_forgetting_removes_a_claim_from_the_remembered_counts() -> None:
    m = memory(2)
    m.record(ints(0), ints(1), tick=0)
    assert m.remembered_counts(ints(1))[0, 0] == 1.0
    m.record(ints(0, 0), ints(2, 3), tick=1)
    assert m.remembered_counts(ints(1))[0, 0] == 0.0


def test_the_empty_write_changes_nothing() -> None:
    m = memory(3, 3)
    m.record(ints(0), ints(1), tick=0)
    claims, written = m.claims.copy(), m.written.copy()
    m.record(ints(), ints(), tick=5)
    assert np.array_equal(m.claims, claims)
    assert np.array_equal(m.written, written)


# --- queries and views ---


def test_recent_rejects_unknown_agents() -> None:
    with pytest.raises(ValueError, match="outside"):
        memory(2).recent(1)


def test_remembered_counts_rejects_unknown_claims() -> None:
    with pytest.raises(ValueError, match="claim index"):
        memory(2, n_claims=5).remembered_counts(ints(5))
    with pytest.raises(ValueError, match="claim index"):
        memory(2, n_claims=5).remembered_counts(ints(-1))


def test_arrays_are_read_only_views() -> None:
    m = memory(2)
    for array in (m.claims, m.ticks, m.written, m.capacity):
        with pytest.raises(ValueError, match="read-only"):
            array[0] = 1


def test_a_copy_is_independent() -> None:
    m = memory(3)
    m.record(ints(0), ints(1), tick=0)
    snapshot = m.copy()
    m.record(ints(0), ints(2), tick=1)
    assert snapshot.recent(0) == [(1, 0)]
    assert m.recent(0) == [(1, 0), (2, 1)]


def test_the_capacity_array_is_not_aliased() -> None:
    capacity = ints(2, 2)
    m = AgentMemory(capacity, 5)
    capacity[0] = 9
    assert m.capacity.tolist() == [2, 2]


# --- validation ---


@pytest.mark.parametrize(
    ("capacity", "n_claims", "error"),
    [
        (ints(-1, 2), 5, ValueError),
        (ints(), 5, ValueError),
        (np.array([1, 2], dtype=np.int32), 5, TypeError),
        (np.array([[1, 2]], dtype=np.int64), 5, ValueError),
        (ints(1), -1, ValueError),
    ],
)
def test_invalid_construction_is_rejected(
    capacity: Any, n_claims: int, error: type[Exception]
) -> None:
    with pytest.raises(error):
        AgentMemory(capacity, n_claims)


@pytest.mark.parametrize(
    ("agents", "claims", "tick", "message"),
    [
        (ints(2), ints(0), 0, "agent index"),
        (ints(0), ints(20), 0, "claim index"),
        (ints(0, 1), ints(0), 0, "same length"),
        (ints(0), ints(0), -1, "tick must be"),
    ],
)
def test_invalid_writes_are_rejected_and_change_nothing(
    agents: Any, claims: Any, tick: int, message: str
) -> None:
    m = memory(2, 2)
    before = (m.claims.copy(), m.written.copy())
    with pytest.raises(ValueError, match=message):
        m.record(agents, claims, tick)
    assert np.array_equal(m.claims, before[0])
    assert np.array_equal(m.written, before[1])


# --- consistency checks ---


def corrupted(mutation: Any) -> AgentMemory:
    m = memory(3, 3)
    m.record(ints(0, 0, 1), ints(1, 2, 3), tick=4)
    mutation(m)
    return m


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda m: m._written.__setitem__(0, -1), "negative"),
        (lambda m: m._capacity.__setitem__(0, 9), "exceeds"),
        (lambda m: m._claims.__setitem__((0, 0), -1), "filled slots"),
        (lambda m: m._claims.__setitem__((1, 1), 5), "filled slots"),
        (lambda m: m._claims.__setitem__((0, 0), 99), "invalid claim or tick"),
        (lambda m: m._ticks.__setitem__((0, 0), -1), "invalid claim or tick"),
        (lambda m: m._ticks.__setitem__((1, 2), 4), "empty slot has a tick"),
    ],
)
def test_validate_detects_each_kind_of_corruption(mutation: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        corrupted(mutation).validate()
