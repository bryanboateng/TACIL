import itertools
from dataclasses import dataclass

import torch
from tqdm import tqdm

from scripts.data.augmentation.tacil.create_data.chain_error import (
    ChainErrorCriterion,
    ChainErrorTracker,
)
from scripts.data.augmentation.tacil.create_data.episode_assembly import (
    SyntheticTrajectories,
)
from scripts.data.augmentation.tacil.create_data.inverse_dynamics import (
    solve_inverse_dynamics,
)
from scripts.data.augmentation.tacil.dynamics.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)


@dataclass(frozen=True)
class BackwardTrajectories:
    predecessor_states: torch.Tensor  # (total_step_count, state_dimensionality)
    predecessor_actions: torch.Tensor  # (total_step_count, action_dimensionality)
    step_count_batch: torch.Tensor  # (anchor_count,)

    @property
    def start_index_batch(self) -> torch.Tensor:  # (anchor_count,)
        return torch.cumsum(input=self.step_count_batch, dim=0) - self.step_count_batch

    def to_temporal_order(self) -> SyntheticTrajectories:
        """Reverse each chain: the walk found predecessors anchor-first, but a
        training episode runs the other way, toward the anchor."""
        step_count_batch = self.step_count_batch
        start_index_batch = self.start_index_batch
        device = step_count_batch.device

        chain_index_per_step = torch.repeat_interleave(
            input=torch.arange(len(step_count_batch), device=device),
            repeats=step_count_batch,
        )  # (total_step_count,)
        position_in_chain = (
            torch.arange(len(chain_index_per_step), device=device)
            - start_index_batch[chain_index_per_step]
        )  # (total_step_count,)
        source_index = (
            start_index_batch[chain_index_per_step]
            + step_count_batch[chain_index_per_step]
            - 1
            - position_in_chain
        )  # (total_step_count,)

        return SyntheticTrajectories(
            states=self.predecessor_states[source_index],
            actions=self.predecessor_actions[source_index],
            step_count_batch=step_count_batch,
        )


@dataclass(frozen=True)
class RolloutDiagnostics:
    anchor_count: int
    not_converged_count: int
    final_chain_errors: torch.Tensor  # (accepted_anchor_count,)


@torch.no_grad()
def roll_out_backward(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    anchor_state_batch: torch.Tensor,  # (anchor_count, state_dimensionality)
    anchor_action_batch: torch.Tensor,  # (anchor_count, action_dimensionality)
    action_noise_std: torch.Tensor,  # (action_dimensionality,)
    chain_error_criterion: ChainErrorCriterion,
    chain_error_limit: float,
    convergence_tolerance: float,
) -> tuple[BackwardTrajectories, RolloutDiagnostics]:
    anchor_count = anchor_state_batch.shape[0]
    device = anchor_state_batch.device

    step_count_batch = torch.zeros(anchor_count, dtype=torch.long, device=device)
    not_converged_count = 0

    chain_error_tracker = ChainErrorTracker(
        criterion=chain_error_criterion,
        dynamics_model_ensemble=dynamics_model_ensemble,
        anchor_count=anchor_count,
        device=device,
    )

    current_state_batch = anchor_state_batch.clone()
    active_trajectory_mask_batch = torch.ones(
        anchor_count, dtype=torch.bool, device=device
    )

    # Every test replays the whole prefix, so the accepted actions are
    # kept. Element d holds the action accepted at depth d + 1; anchors that
    # stopped earlier keep zeros there, which are never read.
    chain_action_batch_per_depth: list[torch.Tensor] = []

    accepted_steps_per_depth: list[_AcceptedSteps] = []

    # Overwritten at every accepted step, so what survives is the last one —
    # the value that came closest to the limit before the chain stopped.
    final_chain_error_batch = torch.zeros(anchor_count, device=device)

    progress_bar = tqdm(desc="backward step", unit="step")
    for depth in itertools.count():
        active_trajectory_indices = active_trajectory_mask_batch.nonzero().squeeze(
            dim=1
        )  # (active_count,)
        if active_trajectory_indices.numel() == 0:
            break
        progress_bar.set_postfix(
            active_anchor_count=int(active_trajectory_indices.numel())
        )

        backward_step = _compute_backward_step(
            dynamics_model_ensemble=dynamics_model_ensemble,
            state_batch=current_state_batch[active_trajectory_indices],
            action_batch=anchor_action_batch[active_trajectory_indices],
            action_noise_std=action_noise_std,
            convergence_tolerance=convergence_tolerance,
        )

        chain_error_batch = chain_error_tracker.measure(
            anchor_indices=active_trajectory_indices,
            predecessor_state_batch=backward_step.predecessor_state_batch,
            predecessor_action_batch=backward_step.predecessor_action_batch,
            successor_state_batch=current_state_batch[active_trajectory_indices],
            anchor_state_batch=anchor_state_batch[active_trajectory_indices],
            earlier_action_batches=[
                chain_action_batch_per_depth[earlier_depth][active_trajectory_indices]
                for earlier_depth in reversed(range(depth))
            ],
        )  # (active_count,)

        accepted_mask_batch = backward_step.has_converged_batch & (
            chain_error_batch <= chain_error_limit
        )  # (active_count,)

        accepted_steps = _AcceptedSteps(
            anchor_indices=active_trajectory_indices[accepted_mask_batch],
            predecessor_state_batch=backward_step.predecessor_state_batch[
                accepted_mask_batch
            ],
            predecessor_action_batch=backward_step.predecessor_action_batch[
                accepted_mask_batch
            ],
        )

        accepted_steps_per_depth.append(accepted_steps)
        step_count_batch[accepted_steps.anchor_indices] = depth + 1

        # Only accepted steps are paid for: a rejected one ends its chain.
        chain_error_tracker.commit(
            anchor_indices=accepted_steps.anchor_indices,
            measured_error_batch=chain_error_batch[accepted_mask_batch],
            predecessor_state_batch=accepted_steps.predecessor_state_batch,
            predecessor_action_batch=accepted_steps.predecessor_action_batch,
        )

        current_state_batch[accepted_steps.anchor_indices] = (
            accepted_steps.predecessor_state_batch
        )

        chain_action_batch = torch.zeros_like(input=anchor_action_batch)
        chain_action_batch[accepted_steps.anchor_indices] = (
            accepted_steps.predecessor_action_batch
        )
        chain_action_batch_per_depth.append(chain_action_batch)

        final_chain_error_batch[accepted_steps.anchor_indices] = chain_error_batch[
            accepted_mask_batch
        ]

        rejected_mask_batch = ~accepted_mask_batch  # (active_count,)
        not_converged_count += int(
            (~backward_step.has_converged_batch[rejected_mask_batch]).sum()
        )
        active_trajectory_mask_batch[active_trajectory_indices[rejected_mask_batch]] = (
            False
        )
        progress_bar.update(n=1)
    progress_bar.close()

    backward_trajectories = _assemble_trajectories(
        accepted_steps_per_depth=accepted_steps_per_depth,
        step_count_batch=step_count_batch,
        state_dimensionality=anchor_state_batch.shape[1],
        action_dimensionality=anchor_action_batch.shape[1],
    )
    return backward_trajectories, RolloutDiagnostics(
        anchor_count=anchor_count,
        not_converged_count=not_converged_count,
        final_chain_errors=final_chain_error_batch[step_count_batch > 0],
    )


# ---- Private --------------------------------------------


@dataclass(frozen=True)
class _AcceptedSteps:
    anchor_indices: torch.Tensor  # (accepted_count,)
    predecessor_state_batch: torch.Tensor  # (accepted_count, state_dimensionality)
    predecessor_action_batch: torch.Tensor  # (accepted_count, action_dimensionality)


@dataclass(frozen=True)
class _BackwardStep:
    predecessor_state_batch: torch.Tensor  # (active_count, state_dimensionality)
    predecessor_action_batch: torch.Tensor  # (active_count, action_dimensionality)
    has_converged_batch: torch.Tensor  # (active_count,)


def _compute_backward_step(
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
    *,
    accepted_steps_per_depth: list[_AcceptedSteps],
    step_count_batch: torch.Tensor,  # (anchor_count,)
    state_dimensionality: int,
    action_dimensionality: int,
) -> BackwardTrajectories:
    device = step_count_batch.device

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
