from typing import Annotated, TypeAlias

import torch
import tyro
from wandb import Run

from scripts.data.augmentation.ccil import augmentation as ccil
from scripts.data.augmentation.ccil.config import CcilConfig
from scripts.data.augmentation.tacil import augmentation as tacil
from scripts.data.augmentation.tacil.augmentation import TacilConfig
from scripts.environments.environment import Environment
from scripts.shared.episodes import Episodes

AugmentationConfig: TypeAlias = (
    Annotated[TacilConfig, tyro.conf.subcommand(name="tacil")]
    | Annotated[CcilConfig, tyro.conf.subcommand(name="ccil")]
)


def get_augmentation_name(*, config: AugmentationConfig | None) -> str | None:
    match config:
        case None:
            return None
        case TacilConfig():
            return "tacil"
        case CcilConfig():
            return "ccil"


def augment(
    *,
    config: AugmentationConfig,
    environment: Environment,
    expert_episodes: Episodes,
    expert_part_length: int,
    seed: int,
    device: torch.device,
    run: Run,
) -> Episodes:
    match config:
        case TacilConfig() as tacil_configuration:
            return tacil.augment(
                config=tacil_configuration,
                environment=environment,
                expert_episodes=expert_episodes,
                expert_part_length=expert_part_length,
                seed=seed,
                device=device,
                run=run,
            )
        case CcilConfig() as ccil_configuration:
            return ccil.augment(
                config=ccil_configuration,
                expert_episodes=expert_episodes,
                expert_part_length=expert_part_length,
                seed=seed,
                device=device,
                run=run,
            )
