import torch

from scripts.data.augmentation.ccil.dynamics_model import DynamicsModel


def solve_inverse_dynamics_state(
    dynamics_model: DynamicsModel,
    initial_state_estimate_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    target_next_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    max_newton_iteration_count: int,
) -> torch.Tensor:  # (batch_size, state_dimensionality)

    ridge_term = 1e-8 * torch.eye(
        n=initial_state_estimate_batch.shape[-1],
        device=initial_state_estimate_batch.device,
        dtype=initial_state_estimate_batch.dtype,
    )

    # Phase 1:
    # Run a non-differentiable Newton solve to find an approximate inverse state.
    # This avoids storing the full optimization trajectory in the autograd graph.
    with torch.no_grad():
        initial_state_estimate_batch_detached = (
            initial_state_estimate_batch.detach().clone()
        )
        action_batch_detached = action_batch.detach()
        target_next_state_batch_detached = target_next_state_batch.detach()

        for _ in range(max_newton_iteration_count):
            next_state_residual_batch: torch.Tensor = (
                dynamics_model.predict_next_state(
                    state_batch=initial_state_estimate_batch_detached,
                    action_batch=action_batch_detached,
                )
                - target_next_state_batch_detached
            )  # (batch_size, state_dimensionality)

            if (next_state_residual_batch.norm(dim=-1) <= 1e-4).all():
                break

            jacobian_wrt_state = _compute_next_state_jacobian_wrt_state(
                dynamics_model=dynamics_model,
                state_batch=initial_state_estimate_batch_detached,
                action_batch=action_batch_detached,
            )

            newton_step_batch: torch.Tensor = torch.linalg.solve(
                A=jacobian_wrt_state + ridge_term,
                B=next_state_residual_batch.unsqueeze(dim=-1),
            ).squeeze(-1)

            initial_state_estimate_batch_detached = (
                initial_state_estimate_batch_detached - newton_step_batch
            )

        solved_state_batch = initial_state_estimate_batch_detached.detach()

    # Phase 2:
    # Freeze the local Jacobian around the converged solution.
    # This gives a stable linearization for the differentiable correction step.
    with torch.no_grad():
        solution_jacobian_batch = _compute_next_state_jacobian_wrt_state(
            dynamics_model=dynamics_model,
            state_batch=solved_state_batch,
            action_batch=action_batch.detach(),
        ).detach()

    # Phase 3:
    # Apply one differentiable Newton correction step.
    # This restores useful gradients without backpropagating through all iterations.
    predicted_next_state_batch = dynamics_model.predict_next_state(
        state_batch=solved_state_batch, action_batch=action_batch
    )

    next_state_residual_batch = predicted_next_state_batch - target_next_state_batch

    newton_step_batch = torch.linalg.solve(
        A=solution_jacobian_batch + ridge_term,
        B=next_state_residual_batch.unsqueeze(-1),
    ).squeeze(-1)

    return solved_state_batch - newton_step_batch


def _compute_next_state_jacobian_wrt_state(
    dynamics_model: DynamicsModel,
    state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
) -> torch.Tensor:  # (batch_size, state_dimensionality, state_dimensionality)

    def next_state(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return dynamics_model.predict_next_state(state_batch=state, action_batch=action)

    return torch.vmap(func=torch.func.jacrev(func=next_state, argnums=0))(
        state_batch, action_batch
    )
