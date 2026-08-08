"""CCIL as one call.

What the policy gets back is the accepted labels alone. A label is a single
(state, action) pair, so there is no episode structure to return.
"""

import torch
from wandb import Run

from scripts.data.augmentation.ccil.config import CcilConfig
from scripts.data.augmentation.ccil.create_augmented_data import (
    create_ccil_augmented_data,
)
from scripts.data.augmentation.ccil.f import TransitionBatch
from scripts.data.base.episode_store import EpisodeStore


def augment(
    *,
    config: CcilConfig,
    expert_episode_store: EpisodeStore,
    seed: int,
    device: torch.device,
    run: Run,
) -> TransitionBatch:
    return create_ccil_augmented_data(
        config=config,
        episode_store=expert_episode_store,
        seed=seed,
        device=device,
        run=run,
    )
