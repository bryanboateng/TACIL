from dataclasses import dataclass
from typing import Protocol

import torch
from tqdm import tqdm

from scripts.data.augmentation.tacil.create_data.config import StoppingRule
from scripts.data.augmentation.tacil.create_data.inverse_dynamics import (
    solve_inverse_dynamics,
)
from scripts.data.augmentation.tacil.dynamics.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)


@dataclass(frozen=True)
class SyntheticTrajectories:
    states: torch.Tensor  # (synthetic_step_count, state_dimensionality)
    actions: torch.Tensor  # (synthetic_step_count, action_dimensionality)
    trajectory_lengths: torch.Tensor  # (trajectory_count,)


@dataclass(frozen=True)
class RolloutDiagnostics:
    anchor_count: int
    not_converged_count: int


def roll_out_backward(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    anchor_state_batch: torch.Tensor,  # (anchor_count, state_dimensionality)
    anchor_action_batch: torch.Tensor,  # (anchor_count, action_dimensionality)
    action_noise_std: torch.Tensor,  # (action_dimensionality,)
    stopping_rule: StoppingRule,
    error_limit: float,
    convergence_tolerance: float,
) -> tuple[SyntheticTrajectories, RolloutDiagnostics]:
    anchor_count = anchor_state_batch.shape[0]
    device = anchor_state_batch.device

    not_converged_count = 0

    error_meter = _build_error_meter(
        stopping_rule=stopping_rule,
        dynamics_model_ensemble=dynamics_model_ensemble,
        anchor_state_batch=anchor_state_batch,
        anchor_action_batch=anchor_action_batch,
    )

    accepted_steps_per_depth: list[_AcceptedSteps] = []

    active_trajectory_indices = torch.arange(
        anchor_count, device=device
    )  # (active_count,)
    current_state_batch = anchor_state_batch  # (active_count, state_dimensionality)

    progress_bar = tqdm(desc="backward step", unit="step")
    while True:
        if active_trajectory_indices.numel() == 0:
            break
        progress_bar.set_postfix(
            active_anchor_count=int(active_trajectory_indices.numel())
        )

        backward_step = _calculate_backward_step(
            dynamics_model_ensemble=dynamics_model_ensemble,
            state_batch=current_state_batch,
            action_batch=anchor_action_batch[active_trajectory_indices],
            action_noise_std=action_noise_std,
            convergence_tolerance=convergence_tolerance,
        )

        error_batch = error_meter.calculate(
            trajectory_indices=active_trajectory_indices, backward_step=backward_step
        )  # (active_count,)

        accepted_mask_batch = backward_step.has_converged_batch & (
            error_batch <= error_limit
        )  # (active_count,)
        not_converged_count += int((~backward_step.has_converged_batch).sum())

        accepted_steps = _AcceptedSteps(
            trajectory_indices=active_trajectory_indices[accepted_mask_batch],
            predecessor_state_batch=backward_step.predecessor_state_batch[
                accepted_mask_batch
            ],
            predecessor_action_batch=backward_step.predecessor_action_batch[
                accepted_mask_batch
            ],
            error_batch=error_batch[accepted_mask_batch],
        )
        accepted_steps_per_depth.append(accepted_steps)
        error_meter.record(accepted_steps=accepted_steps)

        active_trajectory_indices = accepted_steps.trajectory_indices
        current_state_batch = accepted_steps.predecessor_state_batch

        progress_bar.update()
    progress_bar.close()

    synthetic_trajectories = _assemble_trajectories(
        accepted_steps_per_depth=accepted_steps_per_depth, anchor_count=anchor_count
    )
    return synthetic_trajectories, RolloutDiagnostics(
        anchor_count=anchor_count, not_converged_count=not_converged_count
    )


# ---- Private --------------------------------------------


@dataclass(frozen=True)
class _AcceptedSteps:
    trajectory_indices: torch.Tensor  # (accepted_count,)
    predecessor_state_batch: torch.Tensor  # (accepted_count, state_dimensionality)
    predecessor_action_batch: torch.Tensor  # (accepted_count, action_dimensionality)
    error_batch: torch.Tensor  # (accepted_count,)


@dataclass(frozen=True)
class _BackwardStep:
    predecessor_state_batch: torch.Tensor  # (active_count, state_dimensionality)
    predecessor_action_batch: torch.Tensor  # (active_count, action_dimensionality)
    has_converged_batch: torch.Tensor  # (active_count,)


class _ErrorMeter(Protocol):
    def calculate(
        self,
        *,
        trajectory_indices: torch.Tensor,  # (active_count,)
        backward_step: _BackwardStep,
    ) -> torch.Tensor: ...  # (active_count,)

    def record(self, *, accepted_steps: _AcceptedSteps) -> None: ...


class _ReplayMeter:
    def __init__(
        self,
        *,
        dynamics_model_ensemble: DynamicsModelEnsemble,
        anchor_state_batch: torch.Tensor,  # (anchor_count, state_dimensionality)
        anchor_action_batch: torch.Tensor,  # (anchor_count, action_dimensionality)
    ) -> None:
        self._dynamics_model_ensemble = dynamics_model_ensemble
        self._anchor_state_batch = anchor_state_batch
        self._anchor_action_batch = anchor_action_batch

        self._accepted_action_batch_per_depth: list[torch.Tensor] = []

    def calculate(
        self,
        *,
        trajectory_indices: torch.Tensor,  # (active_count,)
        backward_step: _BackwardStep,
    ) -> torch.Tensor:  # (active_count,)
        return self._dynamics_model_ensemble.calculate_replay_error(
            start_state_batch=backward_step.predecessor_state_batch,
            action_batch_list=[
                backward_step.predecessor_action_batch,
                *(
                    accepted_action_batch[trajectory_indices]
                    for accepted_action_batch in reversed(
                        self._accepted_action_batch_per_depth
                    )
                ),
            ],
            end_state_batch=self._anchor_state_batch[trajectory_indices],
        )

    def record(self, *, accepted_steps: _AcceptedSteps) -> None:
        accepted_action_batch = torch.zeros_like(input=self._anchor_action_batch)
        accepted_action_batch[accepted_steps.trajectory_indices] = (
            accepted_steps.predecessor_action_batch
        )
        self._accepted_action_batch_per_depth.append(accepted_action_batch)


class _AccountMeter:
    def __init__(
        self,
        *,
        dynamics_model_ensemble: DynamicsModelEnsemble,
        anchor_count: int,
        device: torch.device,
    ) -> None:
        self._dynamics_model_ensemble = dynamics_model_ensemble
        self._account_batch = torch.zeros(
            anchor_count, device=device
        )  # (anchor_count,)

    def calculate(
        self,
        *,
        trajectory_indices: torch.Tensor,  # (active_count,)
        backward_step: _BackwardStep,
    ) -> torch.Tensor:  # (active_count,)
        return self._account_batch[
            trajectory_indices
        ] + self._dynamics_model_ensemble.calculate_prediction_uncertainty(
            state_batch=backward_step.predecessor_state_batch,
            action_batch=backward_step.predecessor_action_batch,
        )

    def record(self, *, accepted_steps: _AcceptedSteps) -> None:
        self._account_batch[accepted_steps.trajectory_indices] = (
            accepted_steps.error_batch
        )


def _build_error_meter(
    *,
    stopping_rule: StoppingRule,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    anchor_state_batch: torch.Tensor,  # (anchor_count, state_dimensionality)
    anchor_action_batch: torch.Tensor,  # (anchor_count, action_dimensionality)
) -> _ErrorMeter:
    match stopping_rule:
        case "replay":
            return _ReplayMeter(
                dynamics_model_ensemble=dynamics_model_ensemble,
                anchor_state_batch=anchor_state_batch,
                anchor_action_batch=anchor_action_batch,
            )
        case "account":
            return _AccountMeter(
                dynamics_model_ensemble=dynamics_model_ensemble,
                anchor_count=anchor_state_batch.shape[0],
                device=anchor_state_batch.device,
            )


def _calculate_backward_step(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    state_batch: torch.Tensor,  # (active_count, state_dimensionality)
    action_batch: torch.Tensor,  # (active_count, action_dimensionality)
    action_noise_std: torch.Tensor,  # (action_dimensionality,)
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

    return _BackwardStep(
        predecessor_state_batch=inverse_dynamics_solution.state_batch,
        predecessor_action_batch=noisy_action_batch,
        has_converged_batch=inverse_dynamics_solution.converged_batch,
    )


def _assemble_trajectories(
    *, accepted_steps_per_depth: list[_AcceptedSteps], anchor_count: int
) -> SyntheticTrajectories:
    accepted_steps_deepest_first = list(reversed(accepted_steps_per_depth))

    trajectory_index_per_step = torch.cat(
        tensors=[steps.trajectory_indices for steps in accepted_steps_deepest_first]
    )  # (synthetic_step_count,)
    order = torch.argsort(
        input=trajectory_index_per_step, stable=True
    )  # (synthetic_step_count,)

    return SyntheticTrajectories(
        states=torch.cat(
            tensors=[
                steps.predecessor_state_batch for steps in accepted_steps_deepest_first
            ]
        )[order],
        actions=torch.cat(
            tensors=[
                steps.predecessor_action_batch for steps in accepted_steps_deepest_first
            ]
        )[order],
        trajectory_lengths=torch.bincount(
            input=trajectory_index_per_step, minlength=anchor_count
        ),
    )
