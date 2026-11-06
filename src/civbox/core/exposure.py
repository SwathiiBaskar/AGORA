from typing import Final

import numpy as np

from civbox.sim.beliefs import FloatArray, IntArray, require_array

MAX_LEDGER_CELLS: Final = 50_000_000
NEVER: Final = -1
MAX_TICK: Final = 2**31 - 1
MAX_COUNT: Final = 2**32 - 1


def _readonly_view[A: np.ndarray](array: A) -> A:
    view = array.view()
    view.setflags(write=False)
    return view


def check_pairs(
    agents: IntArray, claims: IntArray, n_agents: int, n_claims: int, what: str = "pairs"
) -> None:
    """Raise unless ``agents`` and ``claims`` are equal-length int64 arrays of valid indices."""
    require_array("agents", agents, ndim=1, dtype=np.int64)
    require_array("claims", claims, ndim=1, dtype=np.int64)
    if agents.shape != claims.shape:
        raise ValueError(f"{what}: agents and claims must have the same length")
    if agents.size and (int(agents.min()) < 0 or int(agents.max()) >= n_agents):
        raise ValueError(f"{what}: agent index outside [0, {n_agents})")
    if claims.size and (int(claims.min()) < 0 or int(claims.max()) >= n_claims):
        raise ValueError(f"{what}: claim index outside [0, {n_claims})")


def check_tick(tick: int) -> None:
    if isinstance(tick, bool) or not isinstance(tick, int) or not 0 <= tick <= MAX_TICK:
        raise ValueError(f"tick must be an integer in [0, {MAX_TICK}], got {tick!r}")


class ExposureBuffer:
    """Exposures observed during the current step, in the order they were added.

    Transient: a buffer is scratch space for one step and is not part of any agent's state.
    Order matters only for memory, where the records of one agent are remembered in the
    order they were added.
    """

    def __init__(self, n_agents: int, n_claims: int) -> None:
        self._n_agents = n_agents
        self._n_claims = n_claims
        self._agents: list[IntArray] = []
        self._claims: list[IntArray] = []

    @property
    def n_pending(self) -> int:
        return sum(int(a.size) for a in self._agents)

    def add(self, agents: IntArray, claims: IntArray) -> None:
        """Record that ``agents[i]`` observed ``claims[i]`` for every ``i``."""
        check_pairs(agents, claims, self._n_agents, self._n_claims, "exposure buffer")
        self._agents.append(agents.copy())
        self._claims.append(claims.copy())

    def pairs(self) -> tuple[IntArray, IntArray]:
        """All pending ``(agent, claim)`` pairs, in insertion order."""
        if not self._agents:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty.copy()
        return np.concatenate(self._agents), np.concatenate(self._claims)

    def clear(self) -> None:
        self._agents.clear()
        self._claims.clear()


class ExposureLedger:
    """Persistent record of availability and exposure for every ``(agent, claim)``.

    Mutation happens only through :meth:`make_available` and :meth:`commit`. The arrays are
    exposed as read-only views.
    """

    def __init__(self, n_agents: int, n_claims: int) -> None:
        if n_agents < 1 or n_claims < 0:
            raise ValueError("a ledger needs at least one agent and a non-negative claim count")
        if n_agents * n_claims > MAX_LEDGER_CELLS:
            raise ValueError(
                f"{n_agents} agents x {n_claims} claims exceeds the ledger limit of "
                f"{MAX_LEDGER_CELLS} cells"
            )
        self._n_agents = n_agents
        self._n_claims = n_claims
        shape = (n_agents, n_claims)
        self._available = np.zeros(shape, dtype=np.bool_)
        self._counts = np.zeros(shape, dtype=np.uint32)
        self._first = np.full(shape, NEVER, dtype=np.int32)
        self._last = np.full(shape, NEVER, dtype=np.int32)
        self._tick = NEVER

    @property
    def n_agents(self) -> int:
        return self._n_agents

    @property
    def n_claims(self) -> int:
        return self._n_claims

    @property
    def committed_tick(self) -> int:
        """The tick of the latest commit, or -1 if nothing has been committed."""
        return self._tick

    @property
    def available(self) -> np.ndarray:
        return _readonly_view(self._available)

    @property
    def counts(self) -> np.ndarray:
        """Times each agent has observed each claim (uint32)."""
        return _readonly_view(self._counts)

    @property
    def first_tick(self) -> np.ndarray:
        """Tick of the first exposure, or -1 if never exposed."""
        return _readonly_view(self._first)

    @property
    def last_tick(self) -> np.ndarray:
        """Tick of the latest exposure, or -1 if never exposed."""
        return _readonly_view(self._last)

    def copy(self) -> "ExposureLedger":
        clone = ExposureLedger(self._n_agents, self._n_claims)
        clone._available = self._available.copy()
        clone._counts = self._counts.copy()
        clone._first = self._first.copy()
        clone._last = self._last.copy()
        clone._tick = self._tick
        return clone

    # --- queries ---------------------------------------------------------------------------

    def _check_cell(self, agent: int, claim: int) -> None:
        if not 0 <= agent < self._n_agents:
            raise ValueError(f"agent {agent} outside [0, {self._n_agents})")
        if not 0 <= claim < self._n_claims:
            raise ValueError(f"claim {claim} outside [0, {self._n_claims})")

    def is_available(self, agent: int, claim: int) -> bool:
        self._check_cell(agent, claim)
        return bool(self._available[agent, claim])

    def was_exposed(self, agent: int, claim: int) -> bool:
        """Whether the agent actually observed the claim at least once. Not availability."""
        self._check_cell(agent, claim)
        return bool(self._counts[agent, claim] > 0)

    def counts_for(self, claims: IntArray) -> FloatArray:
        """Observation counts of the given claims as an ``(E, N)`` float array."""
        require_array("claims", claims, ndim=1, dtype=np.int64)
        if claims.size and (int(claims.min()) < 0 or int(claims.max()) >= self._n_claims):
            raise ValueError(f"claim index outside [0, {self._n_claims})")
        selected: FloatArray = self._counts[:, claims].T.astype(np.float64)
        return selected

    # --- transitions -------------------------------------------------------------------------

    def make_available(self, agents: IntArray, claims: IntArray) -> None:
        """Make ``claims[i]`` available to ``agents[i]`` for every ``i``. Idempotent."""
        check_pairs(agents, claims, self._n_agents, self._n_claims, "availability")
        self._available[agents, claims] = True

    def check_commit(self, agents: IntArray, claims: IntArray, tick: int) -> None:
        """Raise unless :meth:`commit` with these arguments would succeed. Changes nothing."""
        check_tick(tick)
        check_pairs(agents, claims, self._n_agents, self._n_claims, "commit")
        if tick < self._tick:
            raise ValueError(f"tick {tick} is earlier than the last committed tick {self._tick}")
        if agents.size == 0:
            return
        missing = np.flatnonzero(~self._available[agents, claims])
        if missing.size:
            first = int(missing[0])
            raise ValueError(
                f"agent {int(agents[first])} cannot observe claim {int(claims[first])}: "
                "it was never made available to that agent"
            )
        flat = agents * self._n_claims + claims
        unique, multiplicity = np.unique(flat, return_counts=True)
        total = self._counts.reshape(-1)[unique].astype(np.int64) + multiplicity
        if int(total.max()) > MAX_COUNT:
            raise ValueError("an observation count would exceed the 32-bit limit")

    def commit(self, agents: IntArray, claims: IntArray, tick: int) -> None:
        """Record that ``agents[i]`` observed ``claims[i]`` at ``tick`` for every ``i``.

        Every pair must have been made available. A pair that appears more than once counts
        more than once. Nothing is changed unless the whole commit is valid, and ``tick``
        may not be earlier than the previous commit's.
        """
        self.check_commit(agents, claims, tick)
        if agents.size:
            flat = agents * self._n_claims + claims
            unique, multiplicity = np.unique(flat, return_counts=True)
            counts = self._counts.reshape(-1)
            previous = counts[unique].astype(np.int64)
            counts[unique] = (previous + multiplicity).astype(np.uint32)
            self._first.reshape(-1)[unique[previous == 0]] = tick
            self._last.reshape(-1)[unique] = tick
        self._tick = tick

    # --- consistency ---------------------------------------------------------------------------

    def validate(self) -> None:
        """Raise ``ValueError`` if any ledger invariant is violated."""
        shape = (self._n_agents, self._n_claims)
        for name, array in (
            ("available", self._available),
            ("counts", self._counts),
            ("first_tick", self._first),
            ("last_tick", self._last),
        ):
            if array.shape != shape:
                raise ValueError(f"invalid state: {name} has shape {array.shape}, expected {shape}")
        seen = self._counts > 0
        if (seen & ~self._available).any():
            raise ValueError("invalid state: a claim was observed that was never available")
        if ((self._first == NEVER) != ~seen).any() or ((self._last == NEVER) != ~seen).any():
            raise ValueError("invalid state: exposure ticks disagree with exposure counts")
        if seen.any():
            first, last = self._first[seen], self._last[seen]
            if (first < 0).any() or (first > last).any() or (last > self._tick).any():
                raise ValueError("invalid state: exposure ticks are out of order")
