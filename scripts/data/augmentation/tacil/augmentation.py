from dataclasses import dataclass

import torch
from wandb import Run

from scripts.data.augmentation.tacil.create_data.config import DataConfig
from scripts.data.augmentation.tacil.create_data.create_augmented_data import (
    create_augmented_data,
)
from scripts.data.augmentation.tacil.create_dynamics.config import DynamicsConfig
from scripts.data.augmentation.tacil.create_dynamics.dynamics_model_ensemble_training import (  # noqa: E501
    train_dynamics_model_ensemble,
)
from scripts.environments.environment import Environment
from scripts.shared.episode_store import EpisodeStore


@dataclass(frozen=True)
class TacilConfig:
    dynamics: DynamicsConfig
    data: DataConfig


def augment(
    *,
    config: TacilConfig,
    environment: Environment,
    expert_episodes: EpisodeStore,
    expert_part_length: int,
    seed: int,
    device: torch.device,
    run: Run,
) -> EpisodeStore:
    dynamics_model_ensemble = train_dynamics_model_ensemble(
        config=config.dynamics,
        state_space=environment.state_space,
        network_input_builder=environment.network_input_builder,
        plot_layout=environment.plot_layout,
        episode_store=expert_episodes,
        seed=seed,
        device=device,
        run=run,
    )

    with torch.no_grad():
        return create_augmented_data(
            config=config.data,
            expert_episodes=expert_episodes,
            dynamics_model_ensemble=dynamics_model_ensemble,
            expert_part_length=expert_part_length,
            seed=seed,
            run=run,
        )
