from dataclasses import dataclass
from typing import Annotated, Literal

import tyro

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
    episode_count: int = 100
    seed: int = 42


def describe_policy(*, policy: Policy) -> str:
    match policy:
        case DiffusionPolicyCheckpoint(
            checkpoint_artifact=artifact_name, sampler=sampler
        ):
            return f"dp ({sampler})   {artifact_name}"
        case BehaviorCloningCheckpoint(checkpoint_artifact=artifact_name):
            return f"bc   {artifact_name}"
        case gpi.Config(dataset_artifact=artifact_name):
            return f"gpi   {artifact_name}"
