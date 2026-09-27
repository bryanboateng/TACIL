import torch
from wandb import Run

from scripts.data.augmentation.ccil.config import CcilConfig
from scripts.data.augmentation.ccil.create_augmented_data import (
    create_ccil_augmented_data,
)
from scripts.shared.episodes import Episodes


def augment(
    *,
    config: CcilConfig,
    expert_episodes: Episodes,
    expert_part_length: int,
    seed: int,
    device: torch.device,
    run: Run,
) -> Episodes:
    return create_ccil_augmented_data(
        config=config,
        expert_episodes=expert_episodes,
        expert_part_length=expert_part_length,
        seed=seed,
        device=device,
        run=run,
    )
