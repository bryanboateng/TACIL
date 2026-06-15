from dataclasses import dataclass
from typing import Annotated

import tyro

from scripts.evaluation.bryan.disturbance import DisturberConfig
from scripts.policies.gpi import config as gpi


@dataclass(frozen=True)
class DiffusionPolicyCheckpoint:
    checkpoint_artifact: str


@dataclass(frozen=True)
class BehaviorCloningCheckpoint:
    checkpoint_artifact: str


@dataclass(frozen=True)
class Config:
    policy: (
        Annotated[DiffusionPolicyCheckpoint, tyro.conf.subcommand(name="dp")]
        | Annotated[BehaviorCloningCheckpoint, tyro.conf.subcommand(name="bc")]
        | Annotated[gpi.Config, tyro.conf.subcommand(name="gpi")]
    )
    disturbance: DisturberConfig
    episode_count: int = 100
    seed: int = 42
