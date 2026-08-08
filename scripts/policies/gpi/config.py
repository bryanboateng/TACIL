"""How the geometric policy is built.

The dataset, the seed and the condition belong to the run, not to the policy.
See scripts/policies/gpi/run.py.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ObservationNoiseConfig:
    initial_std: float = 0.01
    decay_rate: float = 0.995
    minimum_std: float = 0.001


@dataclass(frozen=True)
class RetrievalConfig:
    subset_size: int | None = None


@dataclass(frozen=True)
class GuidanceConfig:
    # λ₁ is always static in the current algorithm; λ₂ supports a dynamic path.
    lambda1: float = 1.0
    fixed_lambda2: float | None = None


@dataclass(frozen=True)
class Config:
    noise: ObservationNoiseConfig = field(default_factory=ObservationNoiseConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)
    action_horizon: int = 8
