"""The state of a population of agents (ADR-005).

State is divided by how long it lives and who may change it:

* **Persistent state** survives from step to step and is what a snapshot captures: the
  agents' fixed traits, their beliefs, the exposure ledger, and their memory. It changes only
  through the explicit transitions of :class:`AgentState`, each of which validates its input
  and leaves the state untouched if it is invalid. Arrays are handed out as read-only views.
* **Derived state** is computed from persistent state whenever it is asked for and is never
  stored, so it cannot go stale (:meth:`AgentState.derived`).
* **Transient state** exists for one step only: the :class:`~civbox.sim.exposure.ExposureBuffer`
  of exposures observed so far. It is not part of the agent state and is discarded when its
  contents are committed.
"""

from dataclasses import dataclass
from typing import Self

import numpy as np

from civbox.sim.beliefs import (
    FloatArray,
    IntArray,
    normalized_entropy,
    require_array,
    validate_beliefs,
)
from civbox.sim.exposure import ExposureBuffer, ExposureLedger
from civbox.sim.memory import AgentMemory
from civbox.sim.population import (
    AgentTraits,
    PopulationProfile,
    sample_initial_beliefs,
    sample_traits,
)
from civbox.sim.rng import RngStreams


@dataclass(frozen=True, slots=True)
class DerivedState:
    """Quantities computed from beliefs at the moment they are requested."""

    uncertainty: FloatArray
    leading_hypothesis: IntArray
    leading_probability: FloatArray


class AgentState:
    """Traits, beliefs, exposure ledger, and memory for ``N`` agents and ``K`` hypotheses."""

    def __init__(
        self, traits: AgentTraits, beliefs: FloatArray, floor: float, n_claims: int
    ) -> None:
        validate_beliefs(beliefs, floor)
        if beliefs.shape[0] != traits.n_agents:
            raise ValueError("beliefs must have one row per agent")
        self._traits = traits
        self._floor = floor
        self._beliefs = beliefs.copy()
        self._exposure = ExposureLedger(traits.n_agents, n_claims)
        self._memory = AgentMemory(traits.memory_capacity, n_claims)

    @classmethod
    def sample(
        cls,
        profile: PopulationProfile,
        *,
        n_agents: int,
        n_hypotheses: int,
        n_claims: int,
        floor: float,
        streams: RngStreams,
    ) -> Self:
        """Draw a population from ``profile`` and return its initial state."""
        traits = sample_traits(profile, n_agents, streams)
        beliefs = sample_initial_beliefs(profile, n_agents, n_hypotheses, floor, streams)
        return cls(traits, beliefs, floor, n_claims)

    # --- shape ----------------------------------------------------------------------------------

    @property
    def traits(self) -> AgentTraits:
        return self._traits

    @property
    def floor(self) -> float:
        return self._floor

    @property
    def n_agents(self) -> int:
        return self._traits.n_agents

    @property
    def n_hypotheses(self) -> int:
        return int(self._beliefs.shape[1])

    @property
    def n_claims(self) -> int:
        return self._exposure.n_claims

    @property
    def committed_tick(self) -> int:
        return self._exposure.committed_tick

    # --- read access ----------------------------------------------------------------------------

    @property
    def beliefs(self) -> np.ndarray:
        view = self._beliefs.view()
        view.setflags(write=False)
        return view

    @property
    def available(self) -> np.ndarray:
        return self._exposure.available

    @property
    def exposure_counts(self) -> np.ndarray:
        return self._exposure.counts

    @property
    def first_exposure_tick(self) -> np.ndarray:
        return self._exposure.first_tick

    @property
    def last_exposure_tick(self) -> np.ndarray:
        return self._exposure.last_tick

    def is_available(self, agent: int, claim: int) -> bool:
        return self._exposure.is_available(agent, claim)

    def was_exposed(self, agent: int, claim: int) -> bool:
        """Whether the agent actually observed the claim. Availability alone is not exposure."""
        return self._exposure.was_exposed(agent, claim)

    def observed_counts(self, claims: IntArray) -> FloatArray:
        """Ground-truth observation counts of ``claims``, ``(E, N)``; never forgotten."""
        return self._exposure.counts_for(claims)

    def remembered_counts(self, claims: IntArray) -> FloatArray:
        """How often each agent still *remembers* observing ``claims``, ``(E, N)``."""
        return self._memory.remembered_counts(claims)

    def recent_memory(self, agent: int) -> list[tuple[int, int]]:
        return self._memory.recent(agent)

    def derived(self) -> DerivedState:
        """Uncertainty and leading hypothesis, computed from the current beliefs."""
        leading = self._beliefs.argmax(axis=1).astype(np.int64)
        return DerivedState(
            uncertainty=normalized_entropy(self._beliefs),
            leading_hypothesis=leading,
            leading_probability=self._beliefs[np.arange(self.n_agents), leading],
        )

    def new_exposure_buffer(self) -> ExposureBuffer:
        """A fresh transient buffer for exposures observed during one step."""
        return ExposureBuffer(self.n_agents, self.n_claims)

    # --- transitions ----------------------------------------------------------------------------

    def replace_beliefs(self, beliefs: FloatArray) -> None:
        """Adopt new beliefs, which must be on the floored simplex for this population."""
        validate_beliefs(beliefs, self._floor)
        if beliefs.shape != self._beliefs.shape:
            raise ValueError(f"beliefs must have shape {self._beliefs.shape}, got {beliefs.shape}")
        self._beliefs = beliefs.copy()

    def make_available(self, agents: IntArray, claims: IntArray) -> None:
        """Make ``claims[i]`` available to ``agents[i]``. Availability is not exposure."""
        self._exposure.make_available(agents, claims)

    def commit_exposures(self, buffer: ExposureBuffer, tick: int) -> None:
        """Make the buffered observations part of the state, then empty the buffer.

        Updates the exposure ledger and each agent's memory together. Raises, changing
        nothing, if any observation is of a claim that was not available to that agent.
        """
        agents, claims = buffer.pairs()
        self._exposure.check_commit(agents, claims, tick)
        self._exposure.commit(agents, claims, tick)
        self._memory.record(agents, claims, tick)
        buffer.clear()

    # --- persistence boundary -------------------------------------------------------------------

    def persistent_arrays(self) -> dict[str, np.ndarray]:
        """Every persistent array, by stable name, as read-only views.

        Derived and transient state are not included. Two states are identical exactly when
        these arrays are equal.
        """
        traits = self._traits.arrays()
        arrays: dict[str, np.ndarray] = {f"trait_{name}": a for name, a in traits.items()}
        arrays["beliefs"] = self.beliefs
        arrays["exposure_available"] = self._exposure.available
        arrays["exposure_count"] = self._exposure.counts
        arrays["exposure_first_tick"] = self._exposure.first_tick
        arrays["exposure_last_tick"] = self._exposure.last_tick
        committed = np.array(self._exposure.committed_tick, dtype=np.int64)
        committed.setflags(write=False)
        arrays["exposure_committed_tick"] = committed
        arrays["memory_claims"] = self._memory.claims
        arrays["memory_ticks"] = self._memory.ticks
        arrays["memory_written"] = self._memory.written
        return arrays

    def copy(self) -> Self:
        """An independent deep copy: later changes to either state do not affect the other."""
        clone = type(self).__new__(type(self))
        clone._traits = self._traits
        clone._floor = self._floor
        clone._beliefs = self._beliefs.copy()
        clone._exposure = self._exposure.copy()
        clone._memory = self._memory.copy()
        return clone

    def validate(self) -> None:
        """Raise ``ValueError`` if any invariant of the state is violated."""
        require_array("beliefs", self._beliefs, ndim=2, dtype=np.float64)
        validate_beliefs(self._beliefs, self._floor)
        self._exposure.validate()
        self._memory.validate()
        if (
            self._exposure.n_agents != self.n_agents
            or self._memory.n_agents != self.n_agents
            or self._beliefs.shape[0] != self.n_agents
        ):
            raise ValueError("invalid state: components disagree on the number of agents")
        if (self._memory.capacity != self._traits.memory_capacity).any():
            raise ValueError("invalid state: memory capacities differ from the agents' traits")
