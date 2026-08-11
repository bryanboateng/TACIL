import torch

from scripts.data.augmentation.tacil.models.dynamics_model import DynamicsModel
from scripts.data.augmentation.tacil.models.normalizers import Normalizers
from scripts.shared.pusht_state import subtract_states


def compute_loss(
    *,
    dynamics_model: DynamicsModel,
    normalizers: Normalizers,
    state_window_batch: torch.Tensor,  # (batch_size, window_length + 1, state_dimensionality)
    action_window_batch: torch.Tensor,  # (batch_size, window_length, action_dimensionality)
) -> torch.Tensor:  # (scalar)

    window_length = action_window_batch.shape[1]

    predicted_state_batches: list[torch.Tensor] = []

    current_state_batch = state_window_batch[
        :, 0, :
    ]  # (window_count, state_dimensionality)

    for timestep_index in range(window_length):
        current_action_batch = action_window_batch[
            :, timestep_index, :
        ]  # (window_count, action_dimensionality)

        current_state_batch = dynamics_model.predict_next_state(
            normalizers=normalizers,
            state_batch=current_state_batch,
            action_batch=current_action_batch,
        )  # (window_count, state_dimensionality)
        predicted_state_batches.append(current_state_batch)

    predicted_state_trajectory_batch = torch.stack(
        tensors=predicted_state_batches, dim=1
    )  # (window_count, window_length, state_dimensionality)

    target_state_trajectory_batch = state_window_batch[
        :, 1 : window_length + 1, :
    ]  # (window_count, window_length, state_dimensionality)

    # Measured in typical one-step motion, not in the spread of the whole
    # state: a dimension that barely moves per step (the block, which only
    # moves on contact) would otherwise cost almost nothing to get wrong, and
    # predicting "stays where it is" would be a good enough answer. This is
    # also the space the network itself predicts in.
    return (
        normalizers.scale_to_step_units(
            state_difference_batch=subtract_states(
                minuend_batch=predicted_state_trajectory_batch,
                subtrahend_batch=target_state_trajectory_batch,
            )
        )
        .square()
        .mean()
    )
