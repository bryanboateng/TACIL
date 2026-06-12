import torch

from scripts.data.augmentation.tacil.dynamics_model import DynamicsModel

CONVERGENCE_TOLERANCE = 1e-4


def solve_inverse_dynamics_state(
    *,
    dynamics_model: DynamicsModel,
    initial_state_estimate_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    target_next_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    max_newton_iteration_count: int,
) -> torch.Tensor:  # (batch_size, state_dimensionality)
    normalized_target_next_state_batch = dynamics_model.normalize_state(
        state_batch=target_next_state_batch
    )  # (batch_size, state_dimensionality)

    normalized_converged_state_batch = _run_newton_iterations_until_convergence(
        dynamics_model=dynamics_model,
        normalized_initial_state_estimate=dynamics_model.normalize_state(
            state_batch=initial_state_estimate_batch
        ),
        action=action_batch,
        normalized_target_next_state_batch=normalized_target_next_state_batch,
        max_iteration_count=max_newton_iteration_count,
    )  # (batch_size, state_dimensionality)

    return dynamics_model.denormalize_state(
        normalized_state_batch=normalized_converged_state_batch
    )  # (batch_size, state_dimensionality)


@torch.no_grad()
def _run_newton_iterations_until_convergence(
    *,
    dynamics_model: DynamicsModel,
    normalized_initial_state_estimate: torch.Tensor,  # (batch_size, state_dimensionality)
    action: torch.Tensor,  # (batch_size, action_dimensionality)
    normalized_target_next_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    max_iteration_count: int,
) -> torch.Tensor:  # (batch_size, state_dimensionality)
    normalized_state_estimate_batch = (
        normalized_initial_state_estimate.clone()
    )  # (batch_size, state_dimensionality)

    for _ in range(max_iteration_count):
        normalized_residual_batch = _compute_normalized_residual(
            dynamics_model=dynamics_model,
            normalized_state_batch=normalized_state_estimate_batch,
            action_batch=action,
            normalized_target_next_state_batch=normalized_target_next_state_batch,
        )  # (batch_size, state_dimensionality)
        if (normalized_residual_batch.norm(dim=-1) <= CONVERGENCE_TOLERANCE).all():
            break

        jacobian_batch = (
            _compute_batched_jacobian_of_normalized_residual_wrt_normalized_state(
                dynamics_model=dynamics_model,
                normalized_state_batch=normalized_state_estimate_batch,
                action_batch=action,
                normalized_target_next_state_batch=normalized_target_next_state_batch,
            )
        )  # (batch_size, state_dimensionality, state_dimensionality)
        newton_step_batch: torch.Tensor = torch.linalg.solve(
            A=_add_ridge(jacobian_batch=jacobian_batch),
            B=normalized_residual_batch.unsqueeze(dim=-1),
        ).squeeze(-1)  # (batch_size, state_dimensionality)
        normalized_state_estimate_batch = (
            normalized_state_estimate_batch - newton_step_batch
        )  # (batch_size, state_dimensionality)

    return normalized_state_estimate_batch


def _compute_batched_jacobian_of_normalized_residual_wrt_normalized_state(
    *,
    dynamics_model: DynamicsModel,
    normalized_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    normalized_target_next_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
) -> torch.Tensor:  # (batch_size, state_dimensionality, state_dimensionality)

    def normalized_residual_for_single_element(
        single_normalized_state: torch.Tensor,
        single_action: torch.Tensor,
        single_normalized_target_next_state: torch.Tensor,
    ) -> torch.Tensor:
        return _compute_normalized_residual(
            dynamics_model=dynamics_model,
            normalized_state_batch=single_normalized_state.unsqueeze(dim=0),
            action_batch=single_action.unsqueeze(dim=0),
            normalized_target_next_state_batch=single_normalized_target_next_state.unsqueeze(
                dim=0
            ),
        ).squeeze(dim=0)

    return torch.vmap(
        func=torch.func.jacrev(func=normalized_residual_for_single_element, argnums=0)
    )(normalized_state_batch, action_batch, normalized_target_next_state_batch)


def _compute_normalized_residual(
    *,
    dynamics_model: DynamicsModel,
    normalized_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    normalized_target_next_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
) -> torch.Tensor:  # (batch_size, state_dimensionality)

    predicted_next_state_batch = dynamics_model.predict_next_state(
        state_batch=dynamics_model.denormalize_state(
            normalized_state_batch=normalized_state_batch
        ),
        action_batch=action_batch,
    )  # (batch_size, state_dimensionality)
    return (
        dynamics_model.normalize_state(state_batch=predicted_next_state_batch)
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
