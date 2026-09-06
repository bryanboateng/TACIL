from dataclasses import dataclass

import torch

from scripts.data.augmentation.tacil.dynamics.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)
from scripts.shared.pusht_state import add_states, subtract_states


@dataclass(frozen=True)
class InverseDynamicsSolution:
    state_batch: torch.Tensor  # (batch_size, state_dimensionality)
    converged_batch: torch.Tensor  # (batch_size,)


@torch.no_grad()
def solve_inverse_dynamics(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    initial_state_estimate_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    target_next_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    convergence_tolerance: float,
) -> InverseDynamicsSolution:

    max_iteration_count = 30

    state_estimate_batch = (
        initial_state_estimate_batch.clone()
    )  # (batch_size, state_dimensionality)

    iteration_count = 0
    while True:
        residual_batch = _calculate_residual(
            dynamics_model_ensemble=dynamics_model_ensemble,
            normalized_state_batch=dynamics_model_ensemble.state_normalizer.normalize(
                value_batch=state_estimate_batch
            ),
            action_batch=action_batch,
            target_next_state_batch=target_next_state_batch,
        )  # (batch_size, state_dimensionality)

        converged_batch = (
            residual_batch.abs().amax(dim=1) <= convergence_tolerance
        )  # (batch_size,)

        if converged_batch.all() or iteration_count == max_iteration_count:
            break

        jacobian_batch = _calculate_jacobian_of_residual_wrt_normalized_state(
            dynamics_model_ensemble=dynamics_model_ensemble,
            normalized_state_batch=dynamics_model_ensemble.state_normalizer.normalize(
                value_batch=state_estimate_batch
            ),
            action_batch=action_batch,
            target_next_state_batch=target_next_state_batch,
        )  # (batch_size, state_dimensionality, state_dimensionality)

        normalized_newton_step_batch: torch.Tensor = torch.linalg.solve(
            A=_add_ridge(jacobian_batch=jacobian_batch),
            B=residual_batch.unsqueeze(dim=-1),
        ).squeeze(-1)  # (batch_size, state_dimensionality)

        state_estimate_batch = add_states(
            first_addend_batch=state_estimate_batch,
            second_addend_batch=-dynamics_model_ensemble.state_normalizer.denormalize_difference(
                normalized_difference_batch=normalized_newton_step_batch
            ),
        )  # (batch_size, state_dimensionality)
        iteration_count += 1

    return InverseDynamicsSolution(
        state_batch=state_estimate_batch, converged_batch=converged_batch
    )


# ---- Private --------------------------------------------


def _calculate_residual(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    normalized_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    target_next_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
) -> torch.Tensor:  # (batch_size, state_dimensionality)
    predicted_next_state_batch = dynamics_model_ensemble.predict_next_state(
        state_batch=dynamics_model_ensemble.state_normalizer.denormalize(
            value_batch=normalized_state_batch
        ),
        action_batch=action_batch,
    )  # (batch_size, state_dimensionality)
    return dynamics_model_ensemble.state_normalizer.normalize_difference(
        difference_batch=subtract_states(
            minuend_batch=predicted_next_state_batch,
            subtrahend_batch=target_next_state_batch,
        )
    )  # (batch_size, state_dimensionality)


def _calculate_jacobian_of_residual_wrt_normalized_state(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    normalized_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    target_next_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
) -> torch.Tensor:  # (batch_size, state_dimensionality, state_dimensionality)

    def calculate_residual(
        single_normalized_state: torch.Tensor,  # (state_dimensionality,)
        single_action: torch.Tensor,  # (action_dimensionality,)
        single_target_next_state: torch.Tensor,  # (state_dimensionality,)
    ) -> torch.Tensor:  # (state_dimensionality,)
        return _calculate_residual(
            dynamics_model_ensemble=dynamics_model_ensemble,
            normalized_state_batch=single_normalized_state.unsqueeze(
                dim=0
            ),  # (1, state_dimensionality)
            action_batch=single_action.unsqueeze(dim=0),  # (1, action_dimensionality)
            target_next_state_batch=single_target_next_state.unsqueeze(
                dim=0
            ),  # (1, state_dimensionality)
        ).squeeze(dim=0)  # (state_dimensionality,)

    return torch.vmap(func=torch.func.jacrev(func=calculate_residual, argnums=0))(
        normalized_state_batch,  # (batch_size, state_dimensionality)
        action_batch,  # (batch_size, action_dimensionality)
        target_next_state_batch,  # (batch_size, state_dimensionality)
    )  # (batch_size, state_dimensionality, state_dimensionality)


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
