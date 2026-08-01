from dataclasses import dataclass

from scripts.evaluation.core.condition import DisturbanceLevel, FloorCondition
from scripts.shared.wandb_config import WandbConfig


@dataclass(frozen=True)
class Config:
    wandb: WandbConfig

    floor_condition: FloorCondition
    disturbance_level: DisturbanceLevel | None

    expert_dataset_artifact: str

    # ---- Swept ----------------------------------------------

    lipschitz_constant_upper_bound: str
    window_length: int
    action_noise_fraction: float
    error_limit_fraction: float

    seed: int = 42
