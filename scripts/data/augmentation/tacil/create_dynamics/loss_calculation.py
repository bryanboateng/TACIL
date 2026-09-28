import torch
from torch.nn.utils import parametrize

from scripts.data.augmentation.tacil.dynamics.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)


def calculate_losses(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    state_window_batch: torch.Tensor,
    # (member_count, batch_size, window_length + 1, state_dimensionality), or
    # (batch_size, window_length + 1, state_dimensionality) for one shared batch
    action_window_batch: torch.Tensor,
    # (member_count, batch_size, window_length, action_dimensionality), or
    # (batch_size, window_length, action_dimensionality) for one shared batch
) -> torch.Tensor:  # (member_count,)

    window_length = action_window_batch.shape[-2]

    predicted_state_batches: list[torch.Tensor] = []

    current_state_batch = state_window_batch[
        ..., 0, :
    ]  # (*batch_shape, state_dimensionality)

    with parametrize.cached():
        for timestep_index in range(window_length):
            current_action_batch = action_window_batch[
                ..., timestep_index, :
            ]  # (*batch_shape, action_dimensionality)

            current_state_batch = dynamics_model_ensemble.predict_next_state_per_member(
                state_batch=current_state_batch, action_batch=current_action_batch
            )  # (member_count, batch_size, state_dimensionality)
            predicted_state_batches.append(current_state_batch)

    predicted_state_trajectory_batch = torch.stack(
        tensors=predicted_state_batches, dim=-2
    )  # (member_count, batch_size, window_length, state_dimensionality)

    target_state_trajectory_batch = state_window_batch[
        ..., 1 : window_length + 1, :
    ]  # (*batch_shape, window_length, state_dimensionality)

    return (
        dynamics_model_ensemble.normalizers.state_displacement.normalize_difference(
            difference_batch=dynamics_model_ensemble.state_space.subtract(
                minuend_batch=predicted_state_trajectory_batch,
                subtrahend_batch=target_state_trajectory_batch,
            )
        )
        .square()
        .mean(dim=(1, 2, 3))
    )
