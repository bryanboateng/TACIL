import itertools
from dataclasses import dataclass, field
from enum import IntEnum

import torch
from tqdm import tqdm

from scripts.data.augmentation.tacil.create_data.config import ErrorSignal
from scripts.data.augmentation.tacil.create_data.inverse_dynamics import (
    solve_inverse_dynamics,
)
from scripts.data.augmentation.tacil.models.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)


class TerminationReason(IntEnum):
    ERROR_LIMIT_EXCEEDED = 0
    NOT_CONVERGED = 1


@dataclass(frozen=True)
class BackwardTrajectories:
    predecessor_states: torch.Tensor  # (total_step_count, state_dimensionality)
    predecessor_actions: torch.Tensor  # (total_step_count, action_dimensionality)
    step_count_batch: torch.Tensor  # (anchor_count,)

    @property
    def start_index_batch(self) -> torch.Tensor:  # (anchor_count,)
        return torch.cumsum(input=self.step_count_batch, dim=0) - self.step_count_batch


@dataclass(frozen=True)
class RolloutDiagnostics:
    termination_reasons_batch: torch.Tensor  # (anchor_count,)

    accumulated_error_estimates: torch.Tensor  # (accepted_step_count,)
    step_error_estimates: torch.Tensor  # (accepted_step_count,)
    uncertainties_about_next_state: torch.Tensor  # (accepted_step_count,)
    sensitivity_scaled_step_distances: torch.Tensor  # (accepted_step_count,)
    local_transition_lipschitz_constants: torch.Tensor  # (accepted_step_count,)


@torch.no_grad()
def roll_out_backward(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    anchor_state_batch: torch.Tensor,  # (anchor_count, state_dimensionality)
    anchor_action_batch: torch.Tensor,  # (anchor_count, action_dimensionality)
    action_noise_std: torch.Tensor,  # (action_dimensionality,)
    error_signal: ErrorSignal,
    error_limit: float,
    convergence_tolerance: float,
) -> tuple[BackwardTrajectories, RolloutDiagnostics]:
    anchor_count = anchor_state_batch.shape[0]
    device = anchor_state_batch.device

    step_count_batch = torch.zeros(anchor_count, dtype=torch.long, device=device)
    termination_reason_batch = torch.zeros(
        anchor_count, dtype=torch.long, device=device
    )

    current_state_batch = anchor_state_batch.clone()
    trajectory_error_batch = torch.zeros(anchor_count, device=device)
    error_amplification_batch = torch.ones(anchor_count, device=device)
    active_mask_batch = torch.ones(anchor_count, dtype=torch.bool, device=device)

    accepted_steps_per_depth: list[_AcceptedSteps] = []
    diagnostics_collector = _DiagnosticsCollector()

    progress_bar = tqdm(desc="backward step", unit="step")
    for depth in itertools.count():
        active_indices = active_mask_batch.nonzero().squeeze(dim=1)  # (active_count,)
        if active_indices.numel() == 0:
            break
        progress_bar.set_postfix(active_anchor_count=int(active_indices.numel()))

        backward_step = _compute_backward_step(
            dynamics_model_ensemble=dynamics_model_ensemble,
            state_batch=current_state_batch[active_indices],
            action_batch=anchor_action_batch[active_indices],
            action_noise_std=action_noise_std,
            error_signal=error_signal,
            convergence_tolerance=convergence_tolerance,
        )

        # An error introduced at this depth passes through every already
        # accepted transition on its way to the anchor, so it needs to be
        # amplified accordingly.
        candidate_trajectory_error_batch = (
            trajectory_error_batch[active_indices]
            + error_amplification_batch[active_indices] * backward_step.error_batch
        )  # (active_count,)

        accepted_mask_batch = backward_step.has_converged_batch & (
            candidate_trajectory_error_batch <= error_limit
        )  # (active_count,)

        accepted_steps = _AcceptedSteps(
            anchor_indices=active_indices[accepted_mask_batch],
            predecessor_state_batch=backward_step.predecessor_state_batch[
                accepted_mask_batch
            ],
            predecessor_action_batch=backward_step.predecessor_action_batch[
                accepted_mask_batch
            ],
            trajectory_error_batch=candidate_trajectory_error_batch[
                accepted_mask_batch
            ],
            local_transition_lipschitz_constant_batch=(
                backward_step.local_transition_lipschitz_constant_batch[
                    accepted_mask_batch
                ]
            ),
        )

        accepted_steps_per_depth.append(accepted_steps)
        step_count_batch[accepted_steps.anchor_indices] = depth + 1

        current_state_batch[accepted_steps.anchor_indices] = (
            accepted_steps.predecessor_state_batch
        )
        trajectory_error_batch[accepted_steps.anchor_indices] = (
            accepted_steps.trajectory_error_batch
        )
        error_amplification_batch[accepted_steps.anchor_indices] *= (
            accepted_steps.local_transition_lipschitz_constant_batch
        )

        diagnostics_collector.add(
            backward_step=backward_step,
            accepted_mask_batch=accepted_mask_batch,
            candidate_error_batch=candidate_trajectory_error_batch,
        )

        rejected_mask_batch = ~accepted_mask_batch  # (active_count,)
        rejected_indices = active_indices[rejected_mask_batch]  # (rejected_count,)
        termination_reason_batch[rejected_indices] = torch.where(
            backward_step.has_converged_batch[rejected_mask_batch],
            int(TerminationReason.ERROR_LIMIT_EXCEEDED),
            int(TerminationReason.NOT_CONVERGED),
        )
        active_mask_batch[rejected_indices] = False
        progress_bar.update(n=1)

    progress_bar.close()

    backward_trajectories = _assemble_trajectories(
        accepted_steps_per_depth=accepted_steps_per_depth,
        step_count_batch=step_count_batch,
        state_dimensionality=anchor_state_batch.shape[1],
        action_dimensionality=anchor_action_batch.shape[1],
        device=device,
    )
    return backward_trajectories, diagnostics_collector.diagnostics(
        termination_reasons_batch=termination_reason_batch
    )


# ---- Private --------------------------------------------


@dataclass(frozen=True)
class _AcceptedSteps:
    anchor_indices: torch.Tensor  # (accepted_count,)
    predecessor_state_batch: torch.Tensor  # (accepted_count, state_dimensionality)
    predecessor_action_batch: torch.Tensor  # (accepted_count, action_dimensionality)
    trajectory_error_batch: torch.Tensor  # (accepted_count,)
    local_transition_lipschitz_constant_batch: torch.Tensor  # (accepted_count,)


@dataclass(frozen=True)
class _BackwardStep:
    predecessor_state_batch: torch.Tensor  # (active_count, state_dimensionality)
    predecessor_action_batch: torch.Tensor  # (active_count, action_dimensionality)
    has_converged_batch: torch.Tensor  # (active_count,)
    error_batch: torch.Tensor  # (active_count,)
    uncertainty_about_next_state_batch: torch.Tensor  # (active_count,)
    sensitivity_scaled_step_distance_batch: torch.Tensor  # (active_count,)
    local_transition_lipschitz_constant_batch: torch.Tensor  # (active_count,)


@dataclass
class _DiagnosticsCollector:
    """Collector for the histogram material, kept out of the walk itself.

    Every accepted step adds one value per quantity. Anchors are not kept
    apart, because the histograms pool them anyway.
    """

    accumulated_errors: list[torch.Tensor] = field(default_factory=list)
    step_errors: list[torch.Tensor] = field(default_factory=list)
    uncertainties_about_next_state: list[torch.Tensor] = field(default_factory=list)
    sensitivity_scaled_step_distances: list[torch.Tensor] = field(default_factory=list)
    local_transition_lipschitz_constants: list[torch.Tensor] = field(
        default_factory=list
    )

    def add(
        self,
        *,
        backward_step: _BackwardStep,
        accepted_mask_batch: torch.Tensor,  # (active_count,)
        candidate_error_batch: torch.Tensor,  # (active_count,)
    ) -> None:
        self.accumulated_errors.append(candidate_error_batch[accepted_mask_batch])
        self.step_errors.append(backward_step.error_batch[accepted_mask_batch])
        self.uncertainties_about_next_state.append(
            backward_step.uncertainty_about_next_state_batch[accepted_mask_batch]
        )
        self.sensitivity_scaled_step_distances.append(
            backward_step.sensitivity_scaled_step_distance_batch[accepted_mask_batch]
        )
        self.local_transition_lipschitz_constants.append(
            backward_step.local_transition_lipschitz_constant_batch[accepted_mask_batch]
        )

    def diagnostics(
        self,
        *,
        termination_reasons_batch: torch.Tensor,  # (anchor_count,)
    ) -> RolloutDiagnostics:
        return RolloutDiagnostics(
            termination_reasons_batch=termination_reasons_batch,
            accumulated_error_estimates=torch.cat(tensors=self.accumulated_errors),
            step_error_estimates=torch.cat(tensors=self.step_errors),
            uncertainties_about_next_state=torch.cat(
                tensors=self.uncertainties_about_next_state
            ),
            sensitivity_scaled_step_distances=torch.cat(
                tensors=self.sensitivity_scaled_step_distances
            ),
            local_transition_lipschitz_constants=torch.cat(
                tensors=self.local_transition_lipschitz_constants
            ),
        )


def _compute_backward_step(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    state_batch: torch.Tensor,  # (active_count, state_dimensionality)
    action_batch: torch.Tensor,  # (active_count, action_dimensionality)
    action_noise_std: torch.Tensor,  # (action_dimensionality,)
    error_signal: ErrorSignal,
    convergence_tolerance: float,
) -> _BackwardStep:
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

    uncertainty_about_next_state_batch = (
        dynamics_model_ensemble.uncertainty_about_next_state(
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

    match error_signal:
        case "uncertainty":
            error_batch = uncertainty_about_next_state_batch  # (active_count,)
        case "sensitivity_scaled_step_distance":
            error_batch = sensitivity_scaled_step_distance_batch  # (active_count,)

    return _BackwardStep(
        predecessor_state_batch=predecessor_state_batch,
        predecessor_action_batch=noisy_action_batch,
        has_converged_batch=inverse_dynamics_solution.converged_batch,
        error_batch=error_batch,
        uncertainty_about_next_state_batch=uncertainty_about_next_state_batch,
        sensitivity_scaled_step_distance_batch=sensitivity_scaled_step_distance_batch,
        local_transition_lipschitz_constant_batch=dynamics_model_ensemble.local_transition_lipschitz_constants(
            state_batch=predecessor_state_batch, action_batch=noisy_action_batch
        ),
    )


def _assemble_trajectories(
    *,
    accepted_steps_per_depth: list[_AcceptedSteps],
    step_count_batch: torch.Tensor,  # (anchor_count,)
    state_dimensionality: int,
    action_dimensionality: int,
    device: torch.device,
) -> BackwardTrajectories:
    total_step_count = int(step_count_batch.sum().item())
    predecessor_states = torch.empty(
        total_step_count, state_dimensionality, device=device
    )
    predecessor_actions = torch.empty(
        total_step_count, action_dimensionality, device=device
    )

    anchor_start_index_batch = (
        torch.cumsum(input=step_count_batch, dim=0) - step_count_batch
    )  # (anchor_count,)

    for depth, accepted_steps in enumerate(iterable=accepted_steps_per_depth):
        write_index_batch = (
            anchor_start_index_batch[accepted_steps.anchor_indices] + depth
        )
        if not (
            write_index_batch.shape[0]
            == accepted_steps.predecessor_state_batch.shape[0]
            == accepted_steps.predecessor_action_batch.shape[0]
        ):
            raise ValueError(
                f"Depth {depth} has {write_index_batch.shape[0]} "
                f"write indices but "
                f"{accepted_steps.predecessor_state_batch.shape[0]} states and "
                f"{accepted_steps.predecessor_action_batch.shape[0]} actions"
            )

        predecessor_states[write_index_batch] = accepted_steps.predecessor_state_batch
        predecessor_actions[write_index_batch] = accepted_steps.predecessor_action_batch

    return BackwardTrajectories(
        predecessor_states=predecessor_states,
        predecessor_actions=predecessor_actions,
        step_count_batch=step_count_batch,
    )
