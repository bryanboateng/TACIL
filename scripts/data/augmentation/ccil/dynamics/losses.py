from dataclasses import dataclass

import torch

from scripts.data.augmentation.ccil.dynamics.config import LossConfig
from scripts.data.augmentation.ccil.dynamics.inverse_dynamics import (
    solve_inverse_dynamics_state,
)
from scripts.data.augmentation.ccil.dynamics_model import DynamicsModel

# Newton convergence tolerance for the inner inverse solve during loss computation.
# Not a tuned knob, so kept as a constant rather than config surface.
_INVERSE_CONVERGENCE_TOLERANCE: float = 1e-4


@dataclass(frozen=True)
class LossBreakdown:
    """Detached, gradient-free scalar components, for logging only.

    Components are kept as 0-dim tensors rather than Python floats so they can be
    stacked and averaged over an epoch and synced to the CPU once at log time,
    instead of forcing a device sync every step.
    """

    total: torch.Tensor
    forward: torch.Tensor
    inverse: torch.Tensor


def compute_loss(
    dynamics_model: DynamicsModel,
    state_window_batch: torch.Tensor,  # (batch_size, window_length + 1, state_dimensionality)
    action_window_batch: torch.Tensor,  # (batch_size, window_length, action_dimensionality)
    config: LossConfig,
) -> tuple[torch.Tensor, LossBreakdown]:

    forward_loss = _compute_forward_loss(
        dynamics_model=dynamics_model,
        state_window_batch=state_window_batch,
        action_window_batch=action_window_batch,
    )

    inverse_loss = _compute_inverse_loss(
        dynamics_model=dynamics_model,
        state_window_batch=state_window_batch,
        action_window_batch=action_window_batch,
        max_newton_iteration_count=config.inverse_max_newton_iteration_count,
        convergence_tolerance=_INVERSE_CONVERGENCE_TOLERANCE,
    )

    total_loss = (
        config.forward_inverse_ratio * forward_loss
        + (1 - config.forward_inverse_ratio) * inverse_loss
    )

    loss_breakdown = LossBreakdown(
        total=total_loss.detach(),
        forward=forward_loss.detach(),
        inverse=inverse_loss.detach(),
    )
    return total_loss, loss_breakdown


def _compute_forward_loss(
    dynamics_model: DynamicsModel,
    state_window_batch: torch.Tensor,  # (batch_size, window_length + 1, state_dimensionality)
    action_window_batch: torch.Tensor,  # (batch_size, window_length, action_dimensionality)
) -> torch.Tensor:
    window_length = action_window_batch.shape[1]

    predicted_state_batches: list[torch.Tensor] = []

    current_state_batch = state_window_batch[
        :, 0, :
    ]  # (batch_size, state_dimensionality)

    for timestep_index in range(window_length):
        current_action_batch = action_window_batch[
            :, timestep_index, :
        ]  # (batch_size, action_dimensionality)

        current_state_batch = dynamics_model.predict_next_state(
            state_batch=current_state_batch, action_batch=current_action_batch
        )
        predicted_state_batches.append(current_state_batch)

    predicted_state_trajectory_batch = torch.stack(
        tensors=predicted_state_batches, dim=1
    )  # (batch_size, window_length, state_dimensionality)

    target_state_trajectory_batch = state_window_batch[
        :, 1 : window_length + 1, :
    ]  # (batch_size, window_length, state_dimensionality)

    # Loss is computed in normalized state space: each dimension's error is divided
    # by its std (means cancel in the difference), so every dimension contributes on
    # equal footing regardless of physical scale. normalize_state broadcasts over the
    # trailing dim, so the (batch, window_length, state_dim) tensors pass through fine.
    return torch.nn.functional.mse_loss(
        input=dynamics_model.normalize_state(predicted_state_trajectory_batch),
        target=dynamics_model.normalize_state(target_state_trajectory_batch),
    )


def _compute_inverse_loss(
    dynamics_model: DynamicsModel,
    state_window_batch: torch.Tensor,  # (batch_size, window_length + 1, state_dimensionality)
    action_window_batch: torch.Tensor,  # (batch_size, window_length, action_dimensionality)
    max_newton_iteration_count: int,
    convergence_tolerance: float,
) -> torch.Tensor:
    window_length = action_window_batch.shape[1]

    reconstructed_previous_state_batches: list[torch.Tensor] = []
    ground_truth_previous_state_batches: list[torch.Tensor] = []

    target_next_state_batch = state_window_batch[
        :, -1, :
    ]  # (batch_size, state_dimensionality)

    for timestep_index in range(window_length):
        current_action_batch = action_window_batch[
            :, -(timestep_index + 1), :
        ]  # (batch_size, action_dimensionality)

        reconstructed_previous_state_batch = solve_inverse_dynamics_state(
            dynamics_model=dynamics_model,
            initial_state_estimate_batch=target_next_state_batch,
            action_batch=current_action_batch,
            target_next_state_batch=target_next_state_batch,
            max_newton_iteration_count=max_newton_iteration_count,
        )  # (batch_size, state_dimensionality)

        reconstructed_previous_state_batches.append(reconstructed_previous_state_batch)

        ground_truth_previous_state_batch = state_window_batch[
            :, -(timestep_index + 2), :
        ]
        ground_truth_previous_state_batches.append(ground_truth_previous_state_batch)

        target_next_state_batch = reconstructed_previous_state_batch

    reconstructed_previous_state_tensor = torch.stack(
        tensors=reconstructed_previous_state_batches, dim=1
    )  # (batch_size, window_length, state_dimensionality)
    ground_truth_previous_state_tensor = torch.stack(
        tensors=ground_truth_previous_state_batches, dim=1
    )  # (batch_size, window_length, state_dimensionality)

    # Normalized state space, same rationale as the forward loss.
    return torch.nn.functional.mse_loss(
        input=dynamics_model.normalize_state(reconstructed_previous_state_tensor),
        target=dynamics_model.normalize_state(ground_truth_previous_state_tensor),
    )
