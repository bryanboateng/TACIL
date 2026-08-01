from dataclasses import dataclass
from typing import Literal

from scripts.evaluation.core.condition import DisturbanceLevel, FloorCondition
from scripts.shared.wandb_config import WandbConfig


@dataclass(frozen=True)
class DimensionalityConfig:
    observation: int = 5
    action: int = 2


@dataclass(frozen=True)
class HorizonLengthConfig:
    observation: int = 2
    prediction: int = 16
    action: int = 8


@dataclass(frozen=True)
class OptimizationConfig:
    batch_size: int = 256
    epoch_count: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    learning_rate_warmup_step_count: int = 500
    ema_power: float = 0.75


@dataclass(frozen=True)
class EvaluationConfig:
    floor_condition: FloorCondition
    disturbance_level: DisturbanceLevel | None

    periodic_interval_epoch_count: int = 20
    episode_count: int = 30
    sampler: Literal["ddim", "ddpm"] = "ddim"


@dataclass(frozen=True)
class Config:
    wandb: WandbConfig

    dataset_artifact: str
    dimensionality: DimensionalityConfig
    horizon_length: HorizonLengthConfig
    optimization: OptimizationConfig
    evaluation: EvaluationConfig
    seed: int = 42
    hidden_channel_counts: tuple[int, ...] = (256, 512, 1024)
