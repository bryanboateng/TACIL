"""TaCIL as one call: train the dynamics ensemble, then roll backward.

The two stages share every parameter that matters, so they are configured
together and never leave this process. What the policy gets back is the
synthetic episodes alone.
"""

from dataclasses import dataclass, field

import torch
from wandb import Run

from scripts.data.augmentation.tacil.create_data import config as data_config
from scripts.data.augmentation.tacil.create_data.create_augmented_data import (
    create_augmented_data,
)
from scripts.data.augmentation.tacil.create_dynamics import config as dynamics_config
from scripts.data.augmentation.tacil.create_dynamics.train_dynamics_model_ensemble import (  # noqa: E501
    train_dynamics_model_ensemble,
)
from scripts.data.base.episode_store import EpisodeStore


@dataclass(frozen=True)
class Config:
    """The two stages' own configs, and nothing of its own.

    Every setting is declared once, where the stage that reads it lives. The
    seed is not here: it belongs to the run, not to the method.
    """

    dynamics: dynamics_config.Config = field(default_factory=dynamics_config.Config)
    data: data_config.Config = field(default_factory=data_config.Config)


def augment(
    *,
    config: Config,
    expert_episode_store: EpisodeStore,
    prediction_horizon: int | None,
    seed: int,
    device: torch.device,
    run: Run,
) -> EpisodeStore | None:
    """The synthetic episodes, or None when the rollout produced nothing."""
    dynamics_model_ensemble = train_dynamics_model_ensemble(
        config=config.dynamics,
        episode_store=expert_episode_store,
        seed=seed,
        device=device,
        run=run,
    )

    return create_augmented_data(
        config=config.data,
        episode_store=expert_episode_store,
        dynamics_model_ensemble=dynamics_model_ensemble,
        prediction_horizon=prediction_horizon,
        seed=seed,
        device=device,
        run=run,
    )
