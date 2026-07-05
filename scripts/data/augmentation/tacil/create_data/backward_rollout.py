from dataclasses import dataclass

import torch
from tqdm import tqdm

from scripts.data.augmentation.tacil.create_data.inverse_dynamics import (
    reached_target_next_state,
    solve_inverse_dynamics_state,
)
from scripts.data.augmentation.tacil.dynamics_ensemble import DynamicsEnsemble


@dataclass(frozen=True)
class BackwardStep:
    noisy_action_batch: torch.Tensor  # (N, action_dimensionality)
    predecessor_state_batch: torch.Tensor  # (N, state_dimensionality)
    converged_batch: torch.Tensor  # (N,)
    error_bound_batch: torch.Tensor  # (N,)
    uncertainty_batch: torch.Tensor  # (N,)


@dataclass(frozen=True)
class BackwardTrajectories:
    anchor_alive_batch: torch.Tensor  # (N,)
    predecessor_states_batch: torch.Tensor  # (N, step_count, state_dimensionality)
    predecessor_actions_batch: torch.Tensor  # (N, step_count, action_dimensionality)
    error_bounds_batch: torch.Tensor  # (N, step_count)
    uncertainties_batch: torch.Tensor  # (N, step_count)


@torch.no_grad()
def roll_out_backward(
    *,
    dynamics_ensemble: DynamicsEnsemble,
    state_batch: torch.Tensor,  # (anchor_count, state_dimensionality)
    action_batch: torch.Tensor,  # (anchor_count, action_dimensionality)
    step_count: int,
    action_noise_std: torch.Tensor,  # (action_dimensionality,)
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
    step_uncertainty_batch = torch.empty(
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
            dynamics_ensemble=dynamics_ensemble,
            state_batch=anchor_state_batch[alive_indices],
            action_batch=action_batch[alive_indices],
            action_noise_std=action_noise_std,
        )

        predecessor_state_batch[alive_indices, step_index] = (
            backward_step.predecessor_state_batch
        )
        predecessor_action_batch[alive_indices, step_index] = (
            backward_step.noisy_action_batch
        )
        step_error_bound_batch[alive_indices, step_index] = (
            backward_step.error_bound_batch
        )
        step_uncertainty_batch[alive_indices, step_index] = (
            backward_step.uncertainty_batch
        )

        alive_mask_batch[alive_indices[~backward_step.converged_batch]] = False
        anchor_state_batch[alive_indices[backward_step.converged_batch]] = (
            backward_step.predecessor_state_batch[backward_step.converged_batch]
        )

    return BackwardTrajectories(
        predecessor_states_batch=predecessor_state_batch,
        predecessor_actions_batch=predecessor_action_batch,
        anchor_alive_batch=alive_mask_batch,
        error_bounds_batch=step_error_bound_batch,
        uncertainties_batch=step_uncertainty_batch,
    )


def _compute_backward_step(
    *,
    dynamics_ensemble: DynamicsEnsemble,
    state_batch: torch.Tensor,  # (N, state_dimensionality)
    action_batch: torch.Tensor,  # (N, action_dimensionality)
    action_noise_std: torch.Tensor,  # (action_dimensionality,)
) -> BackwardStep:
    noisy_action_batch = action_batch + torch.normal(
        mean=0.0, std=action_noise_std.expand_as(action_batch)
    )  # (N, action_dimensionality)

    predecessor_state_batch = solve_inverse_dynamics_state(
        dynamics_ensemble=dynamics_ensemble,
        initial_state_estimate_batch=state_batch,
        action_batch=noisy_action_batch,
        target_next_state_batch=state_batch,
    )  # (N, state_dimensionality)

    converged_batch = _check_inverse_dynamics_convergence(
        predecessor_state_batch=predecessor_state_batch,
        action_batch=noisy_action_batch,
        target_state_batch=state_batch,
        dynamics_ensemble=dynamics_ensemble,
    )  # (N,)

    error_bound_batch = _compute_label_error_bound(
        dynamics_ensemble=dynamics_ensemble,
        state_batch=state_batch,
        predecessor_state_batch=predecessor_state_batch,
        action_batch=noisy_action_batch,
    )  # (N,)

    uncertainty_batch = _compute_ensemble_uncertainty(
        dynamics_ensemble=dynamics_ensemble,
        predecessor_state_batch=predecessor_state_batch,
        action_batch=noisy_action_batch,
    )  # (N,)

    return BackwardStep(
        noisy_action_batch=noisy_action_batch,
        predecessor_state_batch=predecessor_state_batch,
        converged_batch=converged_batch,
        error_bound_batch=error_bound_batch,
        uncertainty_batch=uncertainty_batch,
    )


@torch.no_grad()
def _compute_ensemble_uncertainty(
    *,
    dynamics_ensemble: DynamicsEnsemble,
    predecessor_state_batch: torch.Tensor,  # (N, state_dimensionality)
    action_batch: torch.Tensor,  # (N, action_dimensionality)
) -> torch.Tensor:  # (N,)
    per_member_next_state_batch = dynamics_ensemble.predict_next_state_per_member(
        state_batch=predecessor_state_batch, action_batch=action_batch
    )  # (member_count, N, state_dimensionality)

    member_count, batch_size, state_dimensionality = per_member_next_state_batch.shape
    normalized_per_member_next_state_batch = dynamics_ensemble.normalize_state(
        state_batch=per_member_next_state_batch.reshape(-1, state_dimensionality)
    ).reshape(
        member_count, batch_size, state_dimensionality
    )  # (member_count, N, state_dimensionality)

    per_dimension_disagreement_batch = normalized_per_member_next_state_batch.std(
        dim=0, unbiased=False
    )  # (N, state_dimensionality)

    return torch.linalg.norm(per_dimension_disagreement_batch, dim=-1)  # (N,)


def _check_inverse_dynamics_convergence(
    *,
    predecessor_state_batch: torch.Tensor,  # (N, state_dimensionality)
    action_batch: torch.Tensor,  # (N, action_dimensionality)
    target_state_batch: torch.Tensor,  # (N, state_dimensionality)
    dynamics_ensemble: DynamicsEnsemble,
) -> torch.Tensor:  # (N,)
    predicted_state_batch = dynamics_ensemble.predict_next_state(
        state_batch=predecessor_state_batch, action_batch=action_batch
    )  # (N, state_dimensionality)
    next_state_gap_batch = dynamics_ensemble.normalize_state(
        state_batch=predicted_state_batch
    ) - dynamics_ensemble.normalize_state(
        state_batch=target_state_batch
    )  # (N, state_dimensionality)

    return reached_target_next_state(next_state_gap_batch=next_state_gap_batch)  # (N,)


@torch.no_grad()
def _compute_label_error_bound(
    *,
    dynamics_ensemble: DynamicsEnsemble,
    state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    predecessor_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
) -> torch.Tensor:  # (batch_size,)
    local_lipschitz_constant_batch = dynamics_ensemble.local_lipschitz_constants(
        state_batch=state_batch, action_batch=action_batch
    )  # (batch_size,)

    normalized_step_distance_batch: torch.Tensor = torch.linalg.norm(
        dynamics_ensemble.normalize_state(state_batch=predecessor_state_batch)
        - dynamics_ensemble.normalize_state(state_batch=state_batch),
        dim=-1,
    )  # (batch_size,)

    return (
        local_lipschitz_constant_batch * normalized_step_distance_batch
    )  # (batch_size,)
