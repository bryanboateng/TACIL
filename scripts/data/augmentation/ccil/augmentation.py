import torch
from wandb import Run

from scripts.data.augmentation.ccil.config import CcilConfig
from scripts.data.augmentation.ccil.create_augmented_data import (
    create_ccil_augmented_data,
)
from scripts.shared.episode_store import EpisodeStore
from scripts.shared.transition_batch import TransitionBatch


def augment(
    *,
    config: CcilConfig,
    expert_episodes: EpisodeStore,
    seed: int,
    device: torch.device,
    run: Run,
) -> TransitionBatch:
    return create_ccil_augmented_data(
        config=config,
        expert_episodes=expert_episodes,
        seed=seed,
        device=device,
        run=run,
    )
