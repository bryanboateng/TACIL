from dataclasses import dataclass

import torch
from wandb import Run

from scripts.data.augmentation.distillation.config import DistillationDataConfig
from scripts.data.augmentation.distillation.create_data import create_distillation_data
from scripts.data.augmentation.tacil.create_dynamics.config import DynamicsConfig
from scripts.data.augmentation.tacil.create_dynamics.dynamics_model_ensemble_training import (  # noqa: E501
    train_dynamics_model_ensemble,
)
from scripts.shared.episode_store import EpisodeStore


@dataclass(frozen=True)
class DistillationConfig:
    dynamics: DynamicsConfig
    data: DistillationDataConfig


def augment(
    *,
    config: DistillationConfig,
    expert_episodes: EpisodeStore,
    expert_part_length: int,
    seed: int,
    device: torch.device,
    run: Run,
) -> EpisodeStore:
    # The model is what carries a chain forward: GPI says what to do, and the
    # model says where that lands.
    dynamics_model_ensemble = train_dynamics_model_ensemble(
        config=config.dynamics,
        episode_store=expert_episodes,
        seed=seed,
        device=device,
        run=run,
    )

    return create_distillation_data(
        config=config.data,
        episode_store=expert_episodes,
        dynamics_model_ensemble=dynamics_model_ensemble,
        expert_part_length=expert_part_length,
        seed=seed,
        run=run,
    )
