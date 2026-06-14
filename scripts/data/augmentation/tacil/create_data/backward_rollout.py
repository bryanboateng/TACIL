from dataclasses import dataclass

import torch
from tqdm import tqdm

from scripts.data.augmentation.tacil.create_data.inverse_dynamics import (
    CONVERGENCE_TOLERANCE,
    solve_inverse_dynamics_state,
)
from scripts.data.augmentation.tacil.dynamics_model import DynamicsModel


@dataclass(frozen=True)
class BackwardStep:
    noisy_actions: torch.Tensor  # (N, action_dimensionality)
    predecessor_states: torch.Tensor  # (N, state_dimensionality)
    converged: torch.Tensor  # (N,)
    error_bounds: torch.Tensor  # (N,)


@dataclass(frozen=True)
class BackwardTrajectories:
    anchor_alive: torch.Tensor  # (N,)

    predecessor_states: torch.Tensor  # (N, step_count, state_dimensionality)
    predecessor_actions: torch.Tensor  # (N, step_count, action_dimensionality)
    step_bounds: torch.Tensor  # (N, step_count)


@torch.no_grad()
def roll_out_backward(
    *,
    dynamics_model: DynamicsModel,
    state_batch: torch.Tensor,  # (anchor_count, state_dimensionality)
    action_batch: torch.Tensor,  # (anchor_count, action_dimensionality)
    step_count: int,
    action_noise_std: torch.Tensor,  # (action_dimensionality,)
    max_newton_iteration_count: int,
) -> BackwardTrajectories:
    device = state_batch.device
    anchor_count, state_dimensionality = state_batch.shape
    action_dimensionality = action_batch.shape[1]

    anchor_state_batch = state_batch.clone()  # (anchor_count, state_dimensionality)
    predecessor_state_batch = torch.empty(
        size=(anchor_count, step_count, state_dimensionality), device=device
    )  # (anchor_count, step_count, state_dimensionality)
    predecessor_action_batch = torch.empty(
        size=(anchor_count, step_count, action_dimensionality), device=device
    )  # (anchor_count, step_count, action_dimensionality)
    step_error_bound_batch = torch.empty(
        size=(anchor_count, step_count), device=device
    )  # (anchor_count, step_count)
    alive_mask_batch = torch.ones(
        anchor_count, dtype=torch.bool, device=device
    )  # (anchor_count,)

    for step_index in tqdm(range(step_count), desc="backward step", unit="step"):
        alive_indices = alive_mask_batch.nonzero(as_tuple=False).squeeze(
            1
        )  # (alive_count,)

        backward_step = _compute_backward_step(
            dynamics_model=dynamics_model,
            state_batch=anchor_state_batch[alive_indices],
            action_batch=action_batch[alive_indices],
            action_noise_std=action_noise_std,
            max_newton_iteration_count=max_newton_iteration_count,
        )

        predecessor_state_batch[alive_indices, step_index] = (
            backward_step.predecessor_states
        )
        predecessor_action_batch[alive_indices, step_index] = (
            backward_step.noisy_actions
        )
        step_error_bound_batch[alive_indices, step_index] = backward_step.error_bounds

        alive_mask_batch[alive_indices[~backward_step.converged]] = False
        anchor_state_batch[alive_indices[backward_step.converged]] = (
            backward_step.predecessor_states[backward_step.converged]
        )

    return BackwardTrajectories(
        predecessor_states=predecessor_state_batch,
        predecessor_actions=predecessor_action_batch,
        anchor_alive=alive_mask_batch,
        step_bounds=step_error_bound_batch,
    )


def _compute_backward_step(
    *,
    dynamics_model: DynamicsModel,
    state_batch: torch.Tensor,  # (N, state_dimensionality)
    action_batch: torch.Tensor,  # (N, action_dimensionality)
    action_noise_std: torch.Tensor,  # (action_dimensionality,)
    max_newton_iteration_count: int,
) -> BackwardStep:
    noisy_action_batch = action_batch + torch.normal(
        mean=0.0, std=action_noise_std.expand_as(action_batch)
    )  # (N, action_dimensionality)

    predecessor_state_batch = solve_inverse_dynamics_state(
        dynamics_model=dynamics_model,
        initial_state_estimate_batch=state_batch,
        action_batch=noisy_action_batch,
        target_next_state_batch=state_batch,
        max_newton_iteration_count=max_newton_iteration_count,
    )  # (N, state_dimensionality)

    converged_batch = _check_inverse_dynamics_convergence(
        predecessor_state_batch=predecessor_state_batch,
        action_batch=noisy_action_batch,
        state_batch=state_batch,
        dynamics_model=dynamics_model,
    )  # (N,)

    error_bound_batch = _compute_label_error_bound(
        dynamics_model=dynamics_model,
        anchor_state_batch=state_batch,
        action_batch=noisy_action_batch,
        predecessor_state_batch=predecessor_state_batch,
    )  # (N,)

    return BackwardStep(
        noisy_actions=noisy_action_batch,
        predecessor_states=predecessor_state_batch,
        converged=converged_batch,
        error_bounds=error_bound_batch,
    )


def _check_inverse_dynamics_convergence(
    *,
    predecessor_state_batch: torch.Tensor,  # (N, state_dimensionality)
    action_batch: torch.Tensor,  # (N, action_dimensionality)
    state_batch: torch.Tensor,  # (N, state_dimensionality)
    dynamics_model: DynamicsModel,
) -> torch.Tensor:  # (N,)
    reconstruction_error_batch: torch.Tensor = (
        dynamics_model.normalize_state(
            state_batch=dynamics_model.predict_next_state(
                state_batch=predecessor_state_batch, action_batch=action_batch
            )
        )
        - dynamics_model.normalize_state(state_batch=state_batch)
    ).norm(dim=-1)  # (N,)

    return reconstruction_error_batch <= CONVERGENCE_TOLERANCE  # (N,)


@torch.no_grad()
def _compute_label_error_bound(
    *,
    dynamics_model: DynamicsModel,
    anchor_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    predecessor_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
) -> torch.Tensor:  # (batch_size,)
    local_lipschitz_constant_batch = dynamics_model.local_lipschitz_constants(
        state_batch=anchor_state_batch, action_batch=action_batch
    )  # (batch_size,)

    normalized_step_distance_batch: torch.Tensor = torch.linalg.norm(
        dynamics_model.normalize_state(state_batch=predecessor_state_batch)
        - dynamics_model.normalize_state(state_batch=anchor_state_batch),
        dim=-1,
    )  # (batch_size,)

    return (
        local_lipschitz_constant_batch * normalized_step_distance_batch
    )  # (batch_size,)
