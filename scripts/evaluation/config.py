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

# How many episodes a policy is scored over. Enough that the reward mean is
# steady from one evaluation to the next.
DEFAULT_EPISODE_COUNT = 100


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

    episode_count: int = DEFAULT_EPISODE_COUNT
    seed: int = 42
