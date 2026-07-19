from dataclasses import dataclass
from enum import IntEnum

import torch
from tqdm import tqdm

from scripts.data.augmentation.tacil.create_data.inverse_dynamics import (
    solve_inverse_dynamics,
)
from scripts.data.augmentation.tacil.models.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)
from scripts.data.augmentation.tacil.models.error_model import ErrorModel
from scripts.shared.log import create_logger

logger = create_logger()

# Hard cap so the rollout tensors stay bounded.
MAX_BACKWARD_STEP_COUNT = 50


class TerminationReason(IntEnum):
    ERROR_LIMIT_EXCEEDED = 0
    NOT_CONVERGED = 1
    STEP_CAP_REACHED = 2


@dataclass(frozen=True)
class BackwardTrajectories:
    # The synthetic prefixes, ordered by backward depth: index 0 is the step
    # closest to the anchor.
    predecessor_states_batch: (
        torch.Tensor
    )  # (anchor_count, step_count, state_dimensionality)
    predecessor_actions_batch: (
        torch.Tensor
    )  # (anchor_count, step_count, action_dimensionality)
    step_count_batch: torch.Tensor  # (anchor_count,)


@dataclass(frozen=True)
class RolloutDiagnostics:
    """Information about what happened during the rollout"""

    termination_reasons_batch: torch.Tensor  # (anchor_count,) TerminationReason values

    # Values of every accepted step, flattened across all anchors: histogram
    # material, so no per-anchor alignment is kept.
    accumulated_error_estimates: torch.Tensor  # (accepted_step_count,)
    step_error_estimates: torch.Tensor  # (accepted_step_count,)
    disagreements_about_next_state: torch.Tensor  # (accepted_step_count,)
    sensitivity_scaled_step_distances: torch.Tensor  # (accepted_step_count,)
    local_transition_lipschitz_constants: torch.Tensor  # (accepted_step_count,)


@torch.no_grad()
def roll_out_backward(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    anchor_state_batch: torch.Tensor,  # (anchor_count, state_dimensionality)
    anchor_action_batch: torch.Tensor,  # (anchor_count, action_dimensionality)
    action_noise_std: torch.Tensor,  # (action_dimensionality,)
    error_model: ErrorModel,
    error_limit: float,
    convergence_tolerance: float,
) -> tuple[BackwardTrajectories, RolloutDiagnostics]:
    anchor_count, state_dimensionality = anchor_state_batch.shape
    device = anchor_state_batch.device

    predecessor_states_batch = torch.zeros(
        anchor_count, MAX_BACKWARD_STEP_COUNT, state_dimensionality, device=device
    )
    predecessor_actions_batch = torch.zeros(
        anchor_count,
        MAX_BACKWARD_STEP_COUNT,
        anchor_action_batch.shape[1],
        device=device,
    )
    step_count_batch = torch.zeros(anchor_count, dtype=torch.long, device=device)
    # Anchors still active when the loop ends keep this default reason.
    termination_reason_batch = torch.full(
        size=(anchor_count,),
        fill_value=int(TerminationReason.STEP_CAP_REACHED),
        dtype=torch.long,
        device=device,
    )

    # The walk's rolling state, one entry per anchor.
    current_state_batch = anchor_state_batch.clone()
    accumulated_error_batch = torch.zeros(anchor_count, device=device)
    # Product of the accepted steps' transition Lipschitz constants: how
    # strongly an error introduced at the current depth grows on its way to
    # the anchor.
    error_amplification_batch = torch.ones(anchor_count, device=device)
    active_mask_batch = torch.ones(anchor_count, dtype=torch.bool, device=device)

    # Diagnostics of the accepted steps, collected flat for logging.
    accumulated_error_estimate_chunks: list[torch.Tensor] = []
    step_error_estimate_chunks: list[torch.Tensor] = []
    disagreement_about_next_state_chunks: list[torch.Tensor] = []
    sensitivity_scaled_step_distance_chunks: list[torch.Tensor] = []
    local_transition_lipschitz_constant_chunks: list[torch.Tensor] = []

    # Progress = anchors whose walk has ended. Counting loop iterations against
    # the step cap would be misleading: almost every rollout ends far earlier.
    progress_bar = tqdm(total=anchor_count, desc="anchors finished", unit="anchor")
    for step_index in range(MAX_BACKWARD_STEP_COUNT):
        active_indices = active_mask_batch.nonzero().squeeze(dim=1)  # (active_count,)
        if active_indices.numel() == 0:
            break

        backward_step = _compute_backward_step(
            dynamics_model_ensemble=dynamics_model_ensemble,
            state_batch=current_state_batch[active_indices],
            action_batch=anchor_action_batch[active_indices],
            action_noise_std=action_noise_std,
            error_model=error_model,
            convergence_tolerance=convergence_tolerance,
        )

        # An error introduced at this depth passes through every already
        # accepted transition on its way to the anchor, so it needs to be
        # amplified accordingly.
        candidate_error_batch = (
            accumulated_error_batch[active_indices]
            + error_amplification_batch[active_indices]
            * backward_step.step_error_estimate_batch
        )  # (active_count,)

        accepted_mask_batch = backward_step.converged_batch & (
            candidate_error_batch <= error_limit
        )  # (active_count,)

        accepted_indices = active_indices[accepted_mask_batch]  # (accepted_count,)

        predecessor_states_batch[accepted_indices, step_index] = (
            backward_step.predecessor_state_batch[accepted_mask_batch]
        )
        predecessor_actions_batch[accepted_indices, step_index] = (
            backward_step.action_batch[accepted_mask_batch]
        )
        step_count_batch[accepted_indices] = step_index + 1

        current_state_batch[accepted_indices] = backward_step.predecessor_state_batch[
            accepted_mask_batch
        ]
        accumulated_error_batch[accepted_indices] = candidate_error_batch[
            accepted_mask_batch
        ]
        error_amplification_batch[accepted_indices] *= (
            backward_step.local_transition_lipschitz_constant_batch[accepted_mask_batch]
        )

        accumulated_error_estimate_chunks.append(
            candidate_error_batch[accepted_mask_batch]
        )
        step_error_estimate_chunks.append(
            backward_step.step_error_estimate_batch[accepted_mask_batch]
        )
        disagreement_about_next_state_chunks.append(
            backward_step.disagreement_about_next_state_batch[accepted_mask_batch]
        )
        sensitivity_scaled_step_distance_chunks.append(
            backward_step.sensitivity_scaled_step_distance_batch[accepted_mask_batch]
        )
        local_transition_lipschitz_constant_chunks.append(
            backward_step.local_transition_lipschitz_constant_batch[accepted_mask_batch]
        )

        rejected_indices = active_indices[~accepted_mask_batch]  # (rejected_count,)
        termination_reason_batch[rejected_indices] = torch.where(
            backward_step.converged_batch[~accepted_mask_batch],
            int(TerminationReason.ERROR_LIMIT_EXCEEDED),
            int(TerminationReason.NOT_CONVERGED),
        )
        active_mask_batch[rejected_indices] = False
        progress_bar.update(n=int(rejected_indices.numel()))

    capped_anchor_count = int(active_mask_batch.sum().item())
    progress_bar.update(n=capped_anchor_count)
    progress_bar.close()
    if capped_anchor_count > 0:
        logger.warning(
            "%d anchors were still within the error limit after %d backward steps; "
            "their trajectories were cut by the step cap, not the error limit.",
            capped_anchor_count,
            MAX_BACKWARD_STEP_COUNT,
        )

    reached_step_count = int(step_count_batch.max().item())
    backward_trajectories = BackwardTrajectories(
        predecessor_states_batch=predecessor_states_batch[:, :reached_step_count],
        predecessor_actions_batch=predecessor_actions_batch[:, :reached_step_count],
        step_count_batch=step_count_batch,
    )
    rollout_diagnostics = RolloutDiagnostics(
        termination_reasons_batch=termination_reason_batch,
        accumulated_error_estimates=torch.cat(
            tensors=accumulated_error_estimate_chunks
        ),
        step_error_estimates=torch.cat(tensors=step_error_estimate_chunks),
        disagreements_about_next_state=torch.cat(
            tensors=disagreement_about_next_state_chunks
        ),
        sensitivity_scaled_step_distances=torch.cat(
            tensors=sensitivity_scaled_step_distance_chunks
        ),
        local_transition_lipschitz_constants=torch.cat(
            tensors=local_transition_lipschitz_constant_chunks
        ),
    )
    return backward_trajectories, rollout_diagnostics


# ---- Private --------------------------------------------


@dataclass(frozen=True)
class _BackwardStep:
    """One backward step per active anchor: the predecessor state reached
    under a noised anchor action, plus the quantities the budget check
    needs to judge the step."""

    predecessor_state_batch: torch.Tensor  # (active_count, state_dimensionality)
    action_batch: torch.Tensor  # (active_count, action_dimensionality)
    converged_batch: torch.Tensor  # (active_count,)
    step_error_estimate_batch: torch.Tensor  # (active_count,)
    disagreement_about_next_state_batch: torch.Tensor  # (active_count,)
    sensitivity_scaled_step_distance_batch: torch.Tensor  # (active_count,)
    local_transition_lipschitz_constant_batch: torch.Tensor  # (active_count,)


def _compute_backward_step(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    state_batch: torch.Tensor,  # (active_count, state_dimensionality)
    action_batch: torch.Tensor,  # (active_count, action_dimensionality)
    action_noise_std: torch.Tensor,  # (action_dimensionality,)
    error_model: ErrorModel,
    convergence_tolerance: float,
) -> _BackwardStep:
    """Noise the base action, solve for the predecessor, estimate the step's
    error and its transition Lipschitz constant."""

    noisy_action_batch = action_batch + torch.normal(
        mean=0.0, std=action_noise_std.expand_as(other=action_batch)
    )  # (active_count, action_dimensionality)

    inverse_dynamics_solution = solve_inverse_dynamics(
        dynamics_model_ensemble=dynamics_model_ensemble,
        initial_state_estimate_batch=state_batch,
        action_batch=noisy_action_batch,
        target_next_state_batch=state_batch,
        convergence_tolerance=convergence_tolerance,
    )
    predecessor_state_batch = (
        inverse_dynamics_solution.state_batch
    )  # (active_count, state_dimensionality)

    disagreement_about_next_state_batch = (
        dynamics_model_ensemble.disagreements_about_next_state(
            state_batch=predecessor_state_batch, action_batch=noisy_action_batch
        )
    )  # (active_count,)

    sensitivity_scaled_step_distance_batch = (
        dynamics_model_ensemble.sensitivity_scaled_step_distances(
            state_batch=predecessor_state_batch,
            next_state_batch=state_batch,
            action_batch=noisy_action_batch,
        )
    )  # (active_count,)

    return _BackwardStep(
        predecessor_state_batch=predecessor_state_batch,
        action_batch=noisy_action_batch,
        converged_batch=inverse_dynamics_solution.converged_batch,
        step_error_estimate_batch=error_model.estimate_step_errors(
            disagreement_batch=disagreement_about_next_state_batch,
            sensitivity_scaled_step_distance_batch=sensitivity_scaled_step_distance_batch,
        ),
        disagreement_about_next_state_batch=disagreement_about_next_state_batch,
        sensitivity_scaled_step_distance_batch=sensitivity_scaled_step_distance_batch,
        local_transition_lipschitz_constant_batch=dynamics_model_ensemble.local_transition_lipschitz_constants(
            state_batch=predecessor_state_batch, action_batch=noisy_action_batch
        ),
    )
