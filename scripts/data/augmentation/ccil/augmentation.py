"""CCIL as one call, with the parameters the sweep searches on the surface.

What the policy gets back is the accepted labels alone. A label is a single
(state, action) pair, so there is no episode structure to return.
"""

from dataclasses import dataclass

import torch
from wandb import Run

from scripts.data.augmentation.ccil import config as ccil_config
from scripts.data.augmentation.ccil.create_augmented_data import (
    create_ccil_augmented_data,
)
from scripts.shared.data import TransitionBatch
from scripts.data.base.episode_store import EpisodeStore


@dataclass(frozen=True)
class Config:
    lipschitz_type: ccil_config.LipschitzType = "soft_sampling"
    lipschitz_constraint: float = 2.0
    type: ccil_config.AugmentationType = "noisy_action"
    # Only noisy_action uses this one.
    action_noise_std: float = 1e-5
    rejection_quantile: float = 0.5


def augment(
    *,
    config: Config,
    expert_episode_store: EpisodeStore,
    seed: int,
    device: torch.device,
    run: Run,
) -> TransitionBatch:
    return create_ccil_augmented_data(
        config=ccil_config.Config(
            dynamics=ccil_config.DynamicsConfig(
                lipschitz_type=config.lipschitz_type,
                lipschitz_constraint=config.lipschitz_constraint,
            ),
            augmentation=ccil_config.AugmentationConfig(
                type=config.type,
                action_noise_std=config.action_noise_std,
                rejection_quantile=config.rejection_quantile,
            ),
            seed=seed,
        ),
        episode_store=expert_episode_store,
        device=device,
        run=run,
    )
