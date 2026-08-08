import numpy as np
import torch

from scripts.data.augmentation.tacil.models.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)
from scripts.shared.f import EpisodeStore
from scripts.shared.pusht_state import state_differences


@torch.no_grad()
def calculate_median_step_distance(
    *, episode_store: EpisodeStore, dynamics_model_ensemble: DynamicsModelEnsemble
) -> float:
    """Typical per-step motion of the expert, in normalized state units."""

    device = next(dynamics_model_ensemble.parameters()).device

    state_batch = torch.as_tensor(data=episode_store.states).to(
        device=device
    )  # (total_timestep_count, state_dimensionality)

    # For every consecutive pair of states (s_t, s_{t+1}), compute how far
    # apart they are in space.
    # This is "how much did the state change in one step" for every step across every episode.
    #
    # The shape is (total_timestep_count - 1,).
    # I.e. one distance per consecutive pair of states.
    step_distance_batch = torch.linalg.norm(
        dynamics_model_ensemble.normalizers.scale_to_state_spread_units(
            state_difference_batch=state_differences(
                state_batch=state_batch[1:], other_state_batch=state_batch[:-1]
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
