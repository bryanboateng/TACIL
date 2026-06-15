from dataclasses import dataclass


@dataclass(frozen=True)
class DimensionalityConfig:
    observation: int = 5
    action: int = 2
    hidden: int = 256


@dataclass(frozen=True)
class OptimizationConfig:
    batch_size: int = 64
    epoch_count: int = 100
    learning_rate: float = 1e-3


@dataclass(frozen=True)
class EvaluationConfig:
    periodic_interval_epoch_count: int = 20
    episode_count: int = 30


@dataclass(frozen=True)
class Config:
    dimensionality: DimensionalityConfig
    optimization: OptimizationConfig
    evaluation: EvaluationConfig
    dataset_artifact: str
    seed: int = 42
