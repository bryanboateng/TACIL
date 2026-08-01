from dataclasses import dataclass
from typing import Annotated, Literal

import tyro

from scripts.evaluation.core.condition import DisturbanceLevel, FloorCondition
from scripts.policies.gpi import config as gpi
from scripts.shared.wandb_config import WandbConfig


@dataclass(frozen=True)
class DiffusionPolicyCheckpoint:
    checkpoint_artifact: str
    sampler: Literal["ddim", "ddpm"] = "ddim"


@dataclass(frozen=True)
class BehaviorCloningCheckpoint:
    checkpoint_artifact: str


Policy = DiffusionPolicyCheckpoint | BehaviorCloningCheckpoint | gpi.Config


@dataclass(frozen=True)
class Config:
    wandb: WandbConfig

    policy: (
        Annotated[DiffusionPolicyCheckpoint, tyro.conf.subcommand(name="dp")]
        | Annotated[BehaviorCloningCheckpoint, tyro.conf.subcommand(name="bc")]
        | Annotated[gpi.Config, tyro.conf.subcommand(name="gpi")]
    )

    floor_condition: FloorCondition
    disturbance_level: DisturbanceLevel | None

    episode_count: int = 100
    seed: int = 42
