import numpy as np
import torch

from scripts.data.augmentation.tacil.dynamics.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)
from scripts.shared.episode_store import EpisodeStore


@torch.no_grad()
def calculate_median_step_distance(
    *, episode_store: EpisodeStore, dynamics_model_ensemble: DynamicsModelEnsemble
) -> float:
    device = next(dynamics_model_ensemble.parameters()).device

    state_batch = torch.as_tensor(data=episode_store.states).to(
        device=device
    )  # (total_timestep_count, state_dimensionality)

    # For every consecutive pair of states (s_t, s_{t+1}), compute how far
    # apart they are in space.
    step_distance_batch = torch.linalg.norm(
        dynamics_model_ensemble.state_normalizer.normalize_difference(
            difference_batch=dynamics_model_ensemble.state_space.subtract(
                minuend_batch=state_batch[1:], subtrahend_batch=state_batch[:-1]
            )
        ),
        dim=1,
    )  # (total_timestep_count - 1,)

    # Because all episodes inside the episode store are concatenated into
    # one long tensor, some "consecutive" pairs are not actually consecutive
    # in time.
    # They are the last state of one episode paired with the first state of
    # the next episode.
    # We need to exclude those pairs.
    episode_boundary_mask = np.zeros(shape=len(step_distance_batch), dtype=bool)
    episode_boundary_mask[episode_store.episode_end_indices[:-1] - 1] = True

    return float(
        step_distance_batch[
            ~torch.as_tensor(data=episode_boundary_mask).to(device=device)
        ]
        .median()
        .item()
    )
