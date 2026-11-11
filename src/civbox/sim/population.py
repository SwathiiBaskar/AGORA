"""Heterogeneous populations (ADR-005).

A :class:`PopulationProfile` describes *distributions* of agent traits, not a single
template: sampling it yields agents that differ from one another. It is plain,
validated, serialisable data, so it can be stored and hashed with a configuration.

Every trait is drawn from its own named random stream (see :mod:`civbox.sim.rng`), so
changing the distribution of one trait never changes the values drawn for another.

Traits are the agent parameters the belief update consumes
(:mod:`civbox.sim.evidence`, :mod:`civbox.sim.pooling`):

* ``confirmation_bias`` in ``[0, 1)``;
* ``susceptibility`` in ``[0, 1]``;
* ``confidence_bound`` in ``(0, 1]``;

plus the size of the agent's short-term memory (:mod:`civbox.sim.memory`) and the
distribution of initial beliefs.
"""

from dataclasses import dataclass
from typing import Annotated, Final, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from civbox.sim.beliefs import FloatArray, IntArray, check_floor, floor_beliefs
from civbox.sim.rng import RngStreams

MAX_MEMORY_CAPACITY: Final = 1024

StrictFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]
PositiveFloat = Annotated[float, Field(strict=True, gt=0.0, allow_inf_nan=False)]
Count = Annotated[int, Field(strict=True, ge=0)]


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Constant(_Model):
    """Every agent has the same value."""

    kind: Literal["constant"] = "constant"
    value: StrictFloat

    def bounds(self) -> tuple[float, float]:
        return (self.value, self.value)

    def sample(self, rng: np.random.Generator, n: int) -> FloatArray:
        return np.full(n, self.value)


class Uniform(_Model):
    """Values uniform on ``[low, high]``."""

    kind: Literal["uniform"] = "uniform"
    low: StrictFloat
    high: StrictFloat

    @model_validator(mode="after")
    def _ordered(self) -> "Uniform":
        if self.low > self.high:
            raise ValueError("low must not exceed high")
        return self

    def bounds(self) -> tuple[float, float]:
        return (self.low, self.high)

    def sample(self, rng: np.random.Generator, n: int) -> FloatArray:
        drawn: FloatArray = self.low + (self.high - self.low) * rng.random(n)
        return np.clip(drawn, self.low, self.high)


class Beta(_Model):
    """A Beta(alpha, beta) distribution scaled onto ``[low, high]``."""

    kind: Literal["beta"] = "beta"
    alpha: PositiveFloat
    beta: PositiveFloat
    low: StrictFloat = 0.0
    high: StrictFloat = 1.0

    @model_validator(mode="after")
    def _ordered(self) -> "Beta":
        if self.low > self.high:
            raise ValueError("low must not exceed high")
        return self

    def bounds(self) -> tuple[float, float]:
        return (self.low, self.high)

    def sample(self, rng: np.random.Generator, n: int) -> FloatArray:
        unit = rng.beta(self.alpha, self.beta, n)
        scaled: FloatArray = self.low + (self.high - self.low) * unit
        return np.clip(scaled, self.low, self.high)


class ConstantCount(_Model):
    """Every agent has the same integer value."""

    kind: Literal["constant"] = "constant"
    value: Count

    def bounds(self) -> tuple[int, int]:
        return (self.value, self.value)

    def sample(self, rng: np.random.Generator, n: int) -> IntArray:
        return np.full(n, self.value, dtype=np.int64)


class UniformCount(_Model):
    """Integers uniform on ``[low, high]`` inclusive."""

    kind: Literal["uniform"] = "uniform"
    low: Count
    high: Count

    @model_validator(mode="after")
    def _ordered(self) -> "UniformCount":
        if self.low > self.high:
            raise ValueError("low must not exceed high")
        return self

    def bounds(self) -> tuple[int, int]:
        return (self.low, self.high)

    def sample(self, rng: np.random.Generator, n: int) -> IntArray:
        drawn: IntArray = rng.integers(self.low, self.high + 1, size=n).astype(np.int64)
        return drawn


Distribution = Annotated[Constant | Uniform | Beta, Field(discriminator="kind")]
CountDistribution = Annotated[ConstantCount | UniformCount, Field(discriminator="kind")]


class PopulationProfile(_Model):
    """Distributions of agent traits and of initial beliefs.

    Every field is required: heterogeneity is a modelling decision and is never filled in
    by default. ``prior_concentration`` is the Dirichlet concentration of initial beliefs,
    either one positive number (symmetric) or one per hypothesis; small values give
    confident agents, large values give agents close to the mean belief.
    """

    confirmation_bias: Distribution
    susceptibility: Distribution
    confidence_bound: Distribution
    memory_capacity: CountDistribution
    prior_concentration: PositiveFloat | tuple[PositiveFloat, ...]

    @model_validator(mode="after")
    def _traits_stay_in_their_valid_ranges(self) -> "PopulationProfile":
        low, high = self.confirmation_bias.bounds()
        if low < 0.0 or high >= 1.0:
            raise ValueError("confirmation_bias must lie in [0, 1)")
        low, high = self.susceptibility.bounds()
        if low < 0.0 or high > 1.0:
            raise ValueError("susceptibility must lie in [0, 1]")
        low, high = self.confidence_bound.bounds()
        if low <= 0.0 or high > 1.0:
            raise ValueError("confidence_bound must lie in (0, 1]")
        if self.memory_capacity.bounds()[1] > MAX_MEMORY_CAPACITY:
            raise ValueError(f"memory_capacity must not exceed {MAX_MEMORY_CAPACITY}")
        if isinstance(self.prior_concentration, tuple) and len(self.prior_concentration) < 2:
            raise ValueError(
                "prior_concentration needs at least two entries when given per hypothesis"
            )
        return self


@dataclass(frozen=True, slots=True)
class AgentTraits:
    """The fixed, per-agent parameters of a sampled population. Arrays are read-only."""

    n_agents: int
    confirmation_bias: FloatArray
    susceptibility: FloatArray
    confidence_bound: FloatArray
    memory_capacity: IntArray

    def arrays(self) -> dict[str, FloatArray | IntArray]:
        return {
            "confirmation_bias": self.confirmation_bias,
            "susceptibility": self.susceptibility,
            "confidence_bound": self.confidence_bound,
            "memory_capacity": self.memory_capacity,
        }


def _readonly[A: np.ndarray](array: A) -> A:
    array.setflags(write=False)
    return array


def sample_traits(profile: PopulationProfile, n_agents: int, streams: RngStreams) -> AgentTraits:
    """Draw every agent's traits. Each trait uses its own stream."""
    if n_agents < 1:
        raise ValueError(f"n_agents must be at least 1, got {n_agents}")
    return AgentTraits(
        n_agents=n_agents,
        confirmation_bias=_readonly(
            profile.confirmation_bias.sample(
                streams.generator("population_confirmation_bias"), n_agents
            )
        ),
        susceptibility=_readonly(
            profile.susceptibility.sample(streams.generator("population_susceptibility"), n_agents)
        ),
        confidence_bound=_readonly(
            profile.confidence_bound.sample(
                streams.generator("population_confidence_bound"), n_agents
            )
        ),
        memory_capacity=_readonly(
            profile.memory_capacity.sample(
                streams.generator("population_memory_capacity"), n_agents
            )
        ),
    )


def sample_initial_beliefs(
    profile: PopulationProfile,
    n_agents: int,
    n_hypotheses: int,
    floor: float,
    streams: RngStreams,
) -> FloatArray:
    """Draw ``(n_agents, n_hypotheses)`` initial beliefs on the floored simplex.

    Beliefs are Dirichlet draws, raised to the floor where a draw falls below it.
    """
    check_floor(n_hypotheses, floor)
    concentration = profile.prior_concentration
    if isinstance(concentration, tuple):
        if len(concentration) != n_hypotheses:
            raise ValueError(
                f"prior_concentration has {len(concentration)} entries "
                f"for {n_hypotheses} hypotheses"
            )
        alpha = np.array(concentration)
    else:
        alpha = np.full(n_hypotheses, concentration)
    drawn = streams.generator("population_initial_beliefs").dirichlet(alpha, size=n_agents)
    return floor_beliefs(drawn, floor)
