from dataclasses import dataclass


@dataclass(frozen=True)
class OptimizationConfig:
    batch_size: int = 100
    epoch_count: int = 100
    learning_rate: float = 1e-3


@dataclass(frozen=True)
class EvaluationConfig:
    periodic_interval_epoch_count: int = 20
    episode_count: int = 30


@dataclass(frozen=True)
class Config:
    optimization: OptimizationConfig
    evaluation: EvaluationConfig
    hidden_layer_dimensionality: int = 256
