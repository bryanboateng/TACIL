import torch

from scripts.data.augmentation.tacil.mmmmmmm.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)


def reached_target_next_state(
    *,
    next_state_gap_batch: torch.Tensor,  # (batch_size, state_dimensionality)
) -> torch.Tensor:  # (batch_size,)
    return next_state_gap_batch.abs().amax(dim=1) <= 1e-4


def solve_inverse_dynamics_state(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    initial_state_estimate_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    target_next_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
) -> torch.Tensor:  # (batch_size, state_dimensionality)
    normalized_target_next_state_batch = dynamics_model_ensemble.normalize_state(
        state_batch=target_next_state_batch
    )  # (batch_size, state_dimensionality)

    normalized_converged_state_batch = _run_newton_iterations_until_convergence(
        dynamics_model_ensemble=dynamics_model_ensemble,
        normalized_initial_state_estimate_batch=dynamics_model_ensemble.normalize_state(
            state_batch=initial_state_estimate_batch
        ),
        action_batch=action_batch,
        normalized_target_next_state_batch=normalized_target_next_state_batch,
    )  # (batch_size, state_dimensionality)

    return dynamics_model_ensemble.denormalize_state(
        normalized_state_batch=normalized_converged_state_batch
    )  # (batch_size, state_dimensionality)


@torch.no_grad()
def _run_newton_iterations_until_convergence(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    normalized_initial_state_estimate_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    normalized_target_next_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
) -> torch.Tensor:  # (batch_size, state_dimensionality)
    max_iteration_count = 30
    normalized_state_estimate_batch = (
        normalized_initial_state_estimate_batch.clone()
    )  # (batch_size, state_dimensionality)

    for _ in range(max_iteration_count):
        next_state_gap_batch = _compute_next_state_gap(
            dynamics_model_ensemble=dynamics_model_ensemble,
            normalized_state_batch=normalized_state_estimate_batch,
            action_batch=action_batch,
            normalized_target_next_state_batch=normalized_target_next_state_batch,
        )  # (batch_size, state_dimensionality)
        if reached_target_next_state(next_state_gap_batch=next_state_gap_batch).all():
            break

        jacobian_batch = (
            _compute_batched_jacobian_of_next_state_gap_wrt_normalized_state(
                dynamics_model_ensemble=dynamics_model_ensemble,
                normalized_state_batch=normalized_state_estimate_batch,
                action_batch=action_batch,
                normalized_target_next_state_batch=normalized_target_next_state_batch,
            )
        )  # (batch_size, state_dimensionality, state_dimensionality)

        newton_step_batch: torch.Tensor = torch.linalg.solve(
            A=_add_ridge(jacobian_batch=jacobian_batch),
            B=next_state_gap_batch.unsqueeze(dim=-1),
        ).squeeze(-1)  # (batch_size, state_dimensionality)

        normalized_state_estimate_batch = (
            normalized_state_estimate_batch - newton_step_batch
        )  # (batch_size, state_dimensionality)

    return normalized_state_estimate_batch


def _compute_batched_jacobian_of_next_state_gap_wrt_normalized_state(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    normalized_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    normalized_target_next_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
) -> torch.Tensor:  # (batch_size, state_dimensionality, state_dimensionality)

    def next_state_gap_for_single_element(
        single_normalized_state: torch.Tensor,  # (state_dimensionality,)
        single_action: torch.Tensor,  # (action_dimensionality,)
        single_normalized_target_next_state: torch.Tensor,  # (state_dimensionality,)
    ) -> torch.Tensor:  # (state_dimensionality,)
        return _compute_next_state_gap(
            dynamics_model_ensemble=dynamics_model_ensemble,
            normalized_state_batch=single_normalized_state.unsqueeze(
                dim=0
            ),  # (1, state_dimensionality)
            action_batch=single_action.unsqueeze(dim=0),  # (1, action_dimensionality)
            normalized_target_next_state_batch=single_normalized_target_next_state.unsqueeze(
                dim=0
            ),  # (1, state_dimensionality)
        ).squeeze(dim=0)  # (state_dimensionality,)

    return torch.vmap(
        func=torch.func.jacrev(func=next_state_gap_for_single_element, argnums=0)
    )(
        normalized_state_batch,  # (batch_size, state_dimensionality)
        action_batch,  # (batch_size, action_dimensionality)
        normalized_target_next_state_batch,  # (batch_size, state_dimensionality)
    )  # (batch_size, state_dimensionality, state_dimensionality)


def _compute_next_state_gap(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    normalized_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    normalized_target_next_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
) -> torch.Tensor:  # (batch_size, state_dimensionality)
    predicted_next_state_batch = dynamics_model_ensemble.predict_next_state(
        state_batch=dynamics_model_ensemble.denormalize_state(
            normalized_state_batch=normalized_state_batch
        ),
        action_batch=action_batch,
    )  # (batch_size, state_dimensionality)
    return (
        dynamics_model_ensemble.normalize_state(state_batch=predicted_next_state_batch)
        - normalized_target_next_state_batch
    )  # (batch_size, state_dimensionality)


def _add_ridge(
    *,
    jacobian_batch: torch.Tensor,  # (batch_size, state_dimensionality, state_dimensionality)
) -> torch.Tensor:  # (batch_size, state_dimensionality, state_dimensionality)
    state_dimensionality = jacobian_batch.shape[-1]
    identity = torch.eye(
        n=state_dimensionality, device=jacobian_batch.device, dtype=jacobian_batch.dtype
    )  # (state_dimensionality, state_dimensionality)
    return (
        jacobian_batch + 1e-8 * identity
    )  # (batch_size, state_dimensionality, state_dimensionality)
