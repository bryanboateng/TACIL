import numpy as np
import torch
from numpy import typing as npt

from scripts.data.augmentation.tacil.dynamics.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)
from scripts.data.augmentation.tacil.dynamics.state_distance import (
    calculate_normalized_state_distance,
)
from scripts.shared.episode_store import EpisodeStore


def calculate_median_state_step_distance(
    *, episode_store: EpisodeStore, dynamics_model_ensemble: DynamicsModelEnsemble
) -> float:
    device = next(dynamics_model_ensemble.parameters()).device

    state_batch = torch.as_tensor(data=episode_store.states).to(
        device=device
    )  # (total_timestep_count, state_dimensionality)

    return _median_over_steps(
        distance_batch=calculate_normalized_state_distance(
            state_batch=state_batch[1:],
            other_state_batch=state_batch[:-1],
            state_space=dynamics_model_ensemble.state_space,
            state_normalizer=dynamics_model_ensemble.state_normalizer,
        ),
        episode_end_indices=episode_store.episode_end_indices,
    )


def calculate_median_action_step_distance(*, episode_store: EpisodeStore) -> float:
    action_batch = torch.as_tensor(
        data=episode_store.actions
    )  # (total_timestep_count, action_dimensionality)

    return _median_over_steps(
        distance_batch=torch.linalg.norm(action_batch[1:] - action_batch[:-1], dim=-1),
        episode_end_indices=episode_store.episode_end_indices,
    )


# ---- Private --------------------------------------------


def _median_over_steps(
    *, distance_batch: torch.Tensor, episode_end_indices: npt.NDArray[np.int64]
) -> float:  # (total_timestep_count - 1,)
    # All episodes sit in one long tensor, so some "consecutive" pairs are the
    # last row of one episode next to the first row of the next.
    # Those are dropped.
    episode_boundary_mask = np.zeros(shape=len(distance_batch), dtype=bool)
    episode_boundary_mask[episode_end_indices[:-1] - 1] = True

    return float(
        distance_batch[
            ~torch.as_tensor(data=episode_boundary_mask).to(
                device=distance_batch.device
            )
        ]
        .median()
        .item()
    )
