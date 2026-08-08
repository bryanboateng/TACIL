from dataclasses import dataclass


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
    dataset_artifact: str
    noise: ObservationNoiseConfig
    retrieval: RetrievalConfig
    guidance: GuidanceConfig
    seed: int
    action_horizon: int = 8
