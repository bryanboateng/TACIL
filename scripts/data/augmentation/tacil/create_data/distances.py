import numpy as np
import torch

from scripts.data.augmentation.tacil.dynamics.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)
from scripts.shared.episode_store import EpisodeStore
from scripts.shared.pusht_state import subtract_states


def calculate_state_distances(
    *,
    minuend_batch: torch.Tensor,  # (..., state_dimensionality)
    subtrahend_batch: torch.Tensor,  # (..., state_dimensionality)
    dynamics_model_ensemble: DynamicsModelEnsemble,
) -> torch.Tensor:  # (...,)
    """How far apart two states lie, in the normalized state space. That space
    is what makes a displacement of the block comparable to one of the agent,
    and the angle wraps rather than running away."""
    return torch.linalg.norm(
        dynamics_model_ensemble.state_normalizer.normalize_difference(
            difference_batch=subtract_states(
                minuend_batch=minuend_batch, subtrahend_batch=subtrahend_batch
            )
        ),
        dim=-1,
    )


@torch.no_grad()
def calculate_state_step_distance(
    *, episode_store: EpisodeStore, dynamics_model_ensemble: DynamicsModelEnsemble
) -> float:
    """How far the state moves in one expert step, at the median."""
    device = next(dynamics_model_ensemble.parameters()).device

    state_batch = torch.as_tensor(data=episode_store.states).to(
        device=device
    )  # (total_timestep_count, state_dimensionality)

    return _median_over_steps(
        distance_batch=calculate_state_distances(
            minuend_batch=state_batch[1:],
            subtrahend_batch=state_batch[:-1],
            dynamics_model_ensemble=dynamics_model_ensemble,
        ),
        episode_store=episode_store,
    )


@torch.no_grad()
def calculate_action_step_distance(*, episode_store: EpisodeStore) -> float:
    """How much the expert changes its action from one step to the next, at the
    median. This is the scale a label difference is read against: an invented
    action that differs by less than this differs by less than the expert's own
    labels do between neighbouring states."""
    action_batch = torch.as_tensor(
        data=episode_store.actions
    )  # (total_timestep_count, action_dimensionality)

    return _median_over_steps(
        distance_batch=torch.linalg.norm(action_batch[1:] - action_batch[:-1], dim=-1),
        episode_store=episode_store,
    )


# ---- Private --------------------------------------------


def _median_over_steps(
    *, distance_batch: torch.Tensor, episode_store: EpisodeStore
) -> float:  # (total_timestep_count - 1,)
    # All episodes sit in one long tensor, so some "consecutive" pairs are the
    # last row of one episode next to the first row of the next. Those are not
    # steps and are dropped.
    episode_boundary_mask = np.zeros(shape=len(distance_batch), dtype=bool)
    episode_boundary_mask[episode_store.episode_end_indices[:-1] - 1] = True

    return float(
        distance_batch[
            ~torch.as_tensor(data=episode_boundary_mask).to(
                device=distance_batch.device
            )
        ]
        .median()
        .item()
    )
