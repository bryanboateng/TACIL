"""Builds new episodes by leaving one demonstration for another.

This is the data-space answer to what GPI does at inference time. GPI keeps
every recorded episode in memory and is free to switch between them mid-run,
which is why it covers states no single demonstration reaches. A policy with
fixed weights cannot look anything up, so the switch has to be in its training
data instead.

A junction is taken where the model says a borrowed action lands where the
other episode goes next. Everything else in a stitched episode is recorded
expert data in a new order, so the labels carry the error of one modelled step,
not of a whole invented chain.

What this does not do is teach recovery: every state on a stitched episode is a
state the expert visited. It adds the combinations, and TACIL adds the way
back.
"""

from dataclasses import dataclass

import torch
from wandb import Run

from scripts.data.augmentation.stitching.config import StitchingDataConfig
from scripts.data.augmentation.stitching.create_data import create_stitched_data
from scripts.data.augmentation.tacil.create_dynamics.config import DynamicsConfig
from scripts.data.augmentation.tacil.create_dynamics.dynamics_model_ensemble_training import (  # noqa: E501
    train_dynamics_model_ensemble,
)
from scripts.shared.episode_store import EpisodeStore


@dataclass(frozen=True)
class StitchingConfig:
    dynamics: DynamicsConfig
    data: StitchingDataConfig


def augment(
    *,
    config: StitchingConfig,
    expert_episodes: EpisodeStore,
    seed: int,
    device: torch.device,
    run: Run,
) -> EpisodeStore:
    # The model is only asked one question per junction: does this borrowed
    # action reach that state? One step, so its error never compounds.
    dynamics_model_ensemble = train_dynamics_model_ensemble(
        config=config.dynamics,
        episode_store=expert_episodes,
        seed=seed,
        device=device,
        run=run,
    )

    return create_stitched_data(
        config=config.data,
        episode_store=expert_episodes,
        dynamics_model_ensemble=dynamics_model_ensemble,
        seed=seed,
        run=run,
    )
