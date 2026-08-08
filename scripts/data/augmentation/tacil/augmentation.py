"""TaCIL as one call: train the dynamics ensemble, then roll backward.

The two stages share every parameter that matters, so they are configured
together and never leave this process. What the policy gets back is the
synthetic episodes alone.
"""

from dataclasses import dataclass

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
    # The model predicts this number of steps from its own output.
    window_length: int = 3
    action_noise_fraction: float = 0.1
    # Fraction of the median expert step distance. It caps how far a synthetic
    # prefix may drift from its anchor.
    error_limit_fraction: float = 0.5
    disagreement_weight: float = 0.5

    # None builds the dynamics model without the bounding layer at all, which
    # is a real option rather than a stand-in number.
    lipschitz_constant_upper_bound: float | None = None

    # The dynamics model's shape and ensemble size. Not swept — the sweep would
    # spend budget on them — but reachable here, so an ablation needs no code
    # change.
    ensemble_size: int = 20
    hidden_layer_count: int = 2
    hidden_layer_dimensionality: int = 256


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
        config=dynamics_config.Config(
            seed=seed,
            window_length=config.window_length,
            lipschitz_constant_upper_bound=config.lipschitz_constant_upper_bound,
            ensemble_size=config.ensemble_size,
            hidden_layer_count=config.hidden_layer_count,
            hidden_layer_dimensionality=config.hidden_layer_dimensionality,
        ),
        episode_store=expert_episode_store,
        device=device,
        run=run,
    )

    return create_augmented_data(
        config=data_config.Config(
            action_noise_fraction=config.action_noise_fraction,
            error_limit_fraction=config.error_limit_fraction,
            disagreement_weight=config.disagreement_weight,
            seed=seed,
        ),
        episode_store=expert_episode_store,
        dynamics_model_ensemble=dynamics_model_ensemble,
        prediction_horizon=prediction_horizon,
        device=device,
        run=run,
    )
