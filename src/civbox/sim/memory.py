"""Bounded short-term memory of the evidence agents have observed (ADR-005).

Each agent remembers the claims it observed most recently, in a ring buffer whose size is
that agent's own ``capacity``. When the buffer is full a new record replaces the oldest one,
so an agent forgets. Memory is therefore bounded by the capacities, however long a
simulation runs, and agents with larger capacities recall more.

Memory is what an *agent* remembers; it can forget. It is separate from the
:class:`civbox.sim.exposure.ExposureLedger`, which is the simulation's ground truth about
what was observed and never forgets.

All operations are vectorised over agents. Within one write, the records of one agent are
stored in the order they were supplied; if an agent receives more records than it has room
for, only the most recent ones are kept. Records that would be overwritten are dropped before
writing, so no write target is repeated and the result does not depend on how NumPy orders
repeated indices.
"""

from typing import Final

import numpy as np

from civbox.sim.beliefs import FloatArray, IntArray, require_array
from civbox.sim.exposure import NEVER, check_pairs, check_tick

MAX_CLAIM_INDEX: Final = 2**31 - 1


def _readonly_view[A: np.ndarray](array: A) -> A:
    view = array.view()
    view.setflags(write=False)
    return view


class AgentMemory:
    """Per-agent ring buffers of ``(claim, tick)`` records."""

    def __init__(self, capacity: IntArray, n_claims: int) -> None:
        require_array("capacity", capacity, ndim=1, dtype=np.int64)
        if capacity.size < 1 or (capacity < 0).any():
            raise ValueError("capacity must be non-negative and cover at least one agent")
        if not 0 <= n_claims <= MAX_CLAIM_INDEX:
            raise ValueError(f"n_claims must be in [0, {MAX_CLAIM_INDEX}]")
        self._n_claims = n_claims
        self._capacity = capacity.copy()
        n_agents, width = int(capacity.size), int(capacity.max())
        self._claims = np.full((n_agents, width), NEVER, dtype=np.int32)
        self._ticks = np.full((n_agents, width), NEVER, dtype=np.int32)
        self._written = np.zeros(n_agents, dtype=np.int64)

    @property
    def n_agents(self) -> int:
        return int(self._capacity.size)

    @property
    def width(self) -> int:
        """The size of the largest buffer; the array width. Fixed for the memory's lifetime."""
        return int(self._claims.shape[1])

    @property
    def capacity(self) -> np.ndarray:
        return _readonly_view(self._capacity)

    @property
    def claims(self) -> np.ndarray:
        """Remembered claim indices, ``(N, width)``; -1 marks an empty slot."""
        return _readonly_view(self._claims)

    @property
    def ticks(self) -> np.ndarray:
        """Tick at which each remembered record was written; -1 for an empty slot."""
        return _readonly_view(self._ticks)

    @property
    def written(self) -> np.ndarray:
        """Records each agent has ever stored (including those since forgotten)."""
        return _readonly_view(self._written)

    def size(self) -> IntArray:
        """Number of records each agent currently remembers."""
        remembered: IntArray = np.minimum(self._written, self._capacity)
        return remembered

    def copy(self) -> "AgentMemory":
        clone = AgentMemory(self._capacity, self._n_claims)
        clone._claims = self._claims.copy()
        clone._ticks = self._ticks.copy()
        clone._written = self._written.copy()
        return clone

    # --- queries ---------------------------------------------------------------------------

    def recent(self, agent: int) -> list[tuple[int, int]]:
        """The agent's remembered ``(claim, tick)`` records, oldest first."""
        if not 0 <= agent < self.n_agents:
            raise ValueError(f"agent {agent} outside [0, {self.n_agents})")
        capacity = int(self._capacity[agent])
        written = int(self._written[agent])
        kept = min(written, capacity)
        return [
            (int(self._claims[agent, j % capacity]), int(self._ticks[agent, j % capacity]))
            for j in range(written - kept, written)
        ]

    def remembered_counts(self, claims: IntArray) -> FloatArray:
        """How many times each agent remembers each of the given claims, ``(E, N)`` floats."""
        require_array("claims", claims, ndim=1, dtype=np.int64)
        if claims.size and (int(claims.min()) < 0 or int(claims.max()) >= self._n_claims):
            raise ValueError(f"claim index outside [0, {self._n_claims})")
        counts = np.zeros((claims.size, self.n_agents))
        for e in range(claims.size):
            counts[e] = (self._claims == claims[e]).sum(axis=1)
        return counts

    # --- transition ----------------------------------------------------------------------------

    def record(self, agents: IntArray, claims: IntArray, tick: int) -> None:
        """Store that ``agents[i]`` observed ``claims[i]`` at ``tick``, for every ``i``.

        Agents with capacity zero remember nothing.
        """
        check_tick(tick)
        check_pairs(agents, claims, self.n_agents, self._n_claims, "memory")
        if agents.size == 0:
            return

        order = np.argsort(agents, kind="stable")
        who, what = agents[order], claims[order]
        has_room = self._capacity[who] > 0
        who, what = who[has_room], what[has_room]
        if who.size == 0:
            return

        starts = np.flatnonzero(np.r_[True, who[1:] != who[:-1]])
        group = np.cumsum(np.r_[True, who[1:] != who[:-1]]) - 1
        per_agent = np.diff(np.r_[starts, who.size])
        rank = np.arange(who.size) - starts[group]
        capacity = self._capacity[who]

        # Position the record would take in the ring, and whether a later record of the same
        # agent in this write overwrites it. Overwritten records are never written.
        slot = (self._written[who] + rank) % capacity
        survives = rank >= per_agent[group] - capacity
        self._claims[who[survives], slot[survives]] = what[survives].astype(np.int32)
        self._ticks[who[survives], slot[survives]] = tick
        self._written[who[starts]] += per_agent

    # --- consistency ---------------------------------------------------------------------------

    def validate(self) -> None:
        """Raise ``ValueError`` if any memory invariant is violated."""
        shape = (self.n_agents, self.width)
        if self._claims.shape != shape or self._ticks.shape != shape:
            raise ValueError("invalid state: memory arrays have the wrong shape")
        if (self._written < 0).any() or (self._capacity < 0).any():
            raise ValueError("invalid state: negative memory counters")
        if (self._capacity > self.width).any():
            raise ValueError("invalid state: a capacity exceeds the memory width")
        slot = np.arange(self.width)[None, :]
        expected = slot < np.minimum(self._written, self._capacity)[:, None]
        filled = self._claims != NEVER
        if (filled != expected).any():
            raise ValueError("invalid state: filled slots do not match the records written")
        if (filled & ((self._claims >= self._n_claims) | (self._ticks < 0))).any():
            raise ValueError("invalid state: a remembered record has an invalid claim or tick")
        if (~filled & (self._ticks != NEVER)).any():
            raise ValueError("invalid state: an empty slot has a tick")
