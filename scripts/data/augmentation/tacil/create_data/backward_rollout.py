from dataclasses import dataclass, field
from enum import IntEnum

import torch
from tqdm import tqdm

from scripts.data.augmentation.tacil.create_data.inverse_dynamics import (
    solve_inverse_dynamics,
)
from scripts.data.augmentation.tacil.models.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)


class TerminationReason(IntEnum):
    ERROR_LIMIT_EXCEEDED = 0
    NOT_CONVERGED = 1
    STEP_CAP_REACHED = 2


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

    replay_landing_errors: torch.Tensor  # (accepted_step_count,)


@torch.no_grad()
def roll_out_backward(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    anchor_state_batch: torch.Tensor,  # (anchor_count, state_dimensionality)
    anchor_action_batch: torch.Tensor,  # (anchor_count, action_dimensionality)
    action_noise_std: torch.Tensor,  # (action_dimensionality,)
    error_limit: float,
    convergence_tolerance: float,
    max_step_count: int,
) -> tuple[BackwardTrajectories, RolloutDiagnostics]:
    anchor_count = anchor_state_batch.shape[0]
    device = anchor_state_batch.device

    step_count_batch = torch.zeros(anchor_count, dtype=torch.long, device=device)
    termination_reason_batch = torch.zeros(
        anchor_count, dtype=torch.long, device=device
    )

    current_state_batch = anchor_state_batch.clone()
    active_mask_batch = torch.ones(anchor_count, dtype=torch.bool, device=device)

    # Entry d holds the noisy action accepted at depth d, indexed by anchor.
    # A chain that is active at depth d has accepted every earlier depth, so
    # its rows in all earlier entries are filled.
    noisy_action_per_depth: list[torch.Tensor] = []

    accepted_steps_per_depth: list[_AcceptedSteps] = []
    diagnostics_collector = _DiagnosticsCollector()

    # Unlike the old accumulated error account, the replay error does not have
    # to grow with depth, so the walk needs a hard stop. The cap also bounds
    # the replay cost, which grows with the square of the depth.
    progress_bar = tqdm(desc="backward step", unit="step", total=max_step_count)
    for depth in range(max_step_count):
        active_indices = active_mask_batch.nonzero().squeeze(dim=1)  # (active_count,)
        if active_indices.numel() == 0:
            break
        progress_bar.set_postfix(active_anchor_count=int(active_indices.numel()))

        noisy_action_batch = anchor_action_batch[active_indices] + torch.normal(
            mean=0.0, std=action_noise_std.expand(active_indices.numel(), -1)
        )  # (active_count, action_dimensionality)

        inverse_dynamics_solution = solve_inverse_dynamics(
            dynamics_model_ensemble=dynamics_model_ensemble,
            initial_state_estimate_batch=current_state_batch[active_indices],
            action_batch=noisy_action_batch,
            target_next_state_batch=current_state_batch[active_indices],
            convergence_tolerance=convergence_tolerance,
        )

        # The promise this candidate makes, measured: every member drives the
        # chain's actions forward from the candidate, and the error is how far
        # the members land from the anchor. Re-measured from scratch at every
        # depth, so nothing is estimated or accumulated across steps.
        replay_error_batch = dynamics_model_ensemble.replay_landing_errors(
            start_state_batch=inverse_dynamics_solution.state_batch,
            action_window_batch=_actions_to_anchor(
                candidate_action_batch=noisy_action_batch,
                noisy_action_per_depth=noisy_action_per_depth,
                active_indices=active_indices,
            ),
            target_state_batch=anchor_state_batch[active_indices],
        )  # (active_count,)

        accepted_mask_batch = inverse_dynamics_solution.converged_batch & (
            replay_error_batch <= error_limit
        )  # (active_count,)

        accepted_steps = _AcceptedSteps(
            anchor_indices=active_indices[accepted_mask_batch],
            predecessor_state_batch=inverse_dynamics_solution.state_batch[
                accepted_mask_batch
            ],
            predecessor_action_batch=noisy_action_batch[accepted_mask_batch],
        )

        accepted_steps_per_depth.append(accepted_steps)
        step_count_batch[accepted_steps.anchor_indices] = depth + 1

        current_state_batch[accepted_steps.anchor_indices] = (
            accepted_steps.predecessor_state_batch
        )

        depth_action_batch = torch.zeros(
            anchor_count, anchor_action_batch.shape[1], device=device
        )
        depth_action_batch[accepted_steps.anchor_indices] = (
            accepted_steps.predecessor_action_batch
        )
        noisy_action_per_depth.append(depth_action_batch)

        diagnostics_collector.add(
            replay_error_batch=replay_error_batch[accepted_mask_batch]
        )

        rejected_mask_batch = ~accepted_mask_batch  # (active_count,)
        rejected_indices = active_indices[rejected_mask_batch]  # (rejected_count,)
        termination_reason_batch[rejected_indices] = torch.where(
            inverse_dynamics_solution.converged_batch[rejected_mask_batch],
            int(TerminationReason.ERROR_LIMIT_EXCEEDED),
            int(TerminationReason.NOT_CONVERGED),
        )
        active_mask_batch[rejected_indices] = False
        progress_bar.update(n=1)

    termination_reason_batch[active_mask_batch] = int(
        TerminationReason.STEP_CAP_REACHED
    )
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


@dataclass
class _DiagnosticsCollector:
    """Collector for the histogram material, kept out of the walk itself.

    Every accepted step adds one value. Anchors are not kept apart, because
    the histograms pool them anyway.
    """

    replay_errors: list[torch.Tensor] = field(default_factory=list)

    def add(self, *, replay_error_batch: torch.Tensor) -> None:
        self.replay_errors.append(replay_error_batch)

    def diagnostics(
        self,
        *,
        termination_reasons_batch: torch.Tensor,  # (anchor_count,)
    ) -> RolloutDiagnostics:
        return RolloutDiagnostics(
            termination_reasons_batch=termination_reasons_batch,
            replay_landing_errors=torch.cat(tensors=self.replay_errors),
        )


def _actions_to_anchor(
    *,
    candidate_action_batch: torch.Tensor,  # (active_count, action_dimensionality)
    noisy_action_per_depth: list[torch.Tensor],  # depth × (anchor_count, action_dim)
    active_indices: torch.Tensor,  # (active_count,)
) -> torch.Tensor:  # (active_count, depth + 1, action_dimensionality)
    """The actions that lead from the candidate predecessor to the anchor, in
    execution order: the candidate's own action first, then the earlier
    accepted actions, newest to oldest."""
    return torch.stack(
        tensors=[
            candidate_action_batch,
            *(
                depth_action_batch[active_indices]
                for depth_action_batch in reversed(noisy_action_per_depth)
            ),
        ],
        dim=1,
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
