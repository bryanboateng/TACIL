import itertools
from dataclasses import dataclass

import torch
from tqdm import tqdm

from scripts.data.augmentation.tacil.create_data.config import StoppingRule
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
class RolloutDiagnostics:
    anchor_count: int
    not_converged_count: int
    final_errors: torch.Tensor  # (accepted_anchor_count,)


@torch.no_grad()
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

    trajectory_length_batch = torch.zeros(anchor_count, dtype=torch.long, device=device)
    not_converged_count = 0

    # What the accepted steps of each trajectory have cost so far. Only the account
    # rule keeps one; replay reads the whole trajectory afresh every time.
    spent_batch = torch.zeros(anchor_count, device=device)

    current_state_batch = anchor_state_batch.clone()
    active_trajectory_mask_batch = torch.ones(
        anchor_count, dtype=torch.bool, device=device
    )

    # Every test replays the whole prefix, so the accepted actions are
    # kept. Element d holds the action accepted at depth d + 1; anchors that
    # stopped earlier keep zeros there, which are never read.
    trajectory_action_batch_per_depth: list[torch.Tensor] = []

    accepted_steps_per_depth: list[_AcceptedSteps] = []

    # Overwritten at every accepted step, so what survives is the last one —
    # the value that came closest to the limit before the trajectory stopped.
    final_error_batch = torch.zeros(anchor_count, device=device)

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

        # The two rules ask different things of the trajectory, which is why they
        # part here: replaying needs the whole action sequence back to the
        # anchor, while the account only ever looks at the newest step.
        if stopping_rule == "replay":
            error_batch = dynamics_model_ensemble.measure_replay_error(
                start_state_batch=backward_step.predecessor_state_batch,
                action_batch_list=[
                    backward_step.predecessor_action_batch,
                    *(
                        trajectory_action_batch_per_depth[earlier_depth][
                            active_trajectory_indices
                        ]
                        for earlier_depth in reversed(range(depth))
                    ),
                ],
                end_state_batch=anchor_state_batch[active_trajectory_indices],
            )  # (active_count,)
        else:
            error_batch = spent_batch[
                active_trajectory_indices
            ] + dynamics_model_ensemble.measure_prediction_disagreement(
                state_batch=backward_step.predecessor_state_batch,
                action_batch=backward_step.predecessor_action_batch,
            )  # (active_count,)

        accepted_mask_batch = backward_step.has_converged_batch & (
            error_batch <= error_limit
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
        trajectory_length_batch[accepted_steps.anchor_indices] = depth + 1

        # Only accepted steps are paid for: a rejected one ends its trajectory, so
        # what it would have cost is never charged.
        spent_batch[accepted_steps.anchor_indices] = error_batch[accepted_mask_batch]

        current_state_batch[accepted_steps.anchor_indices] = (
            accepted_steps.predecessor_state_batch
        )

        trajectory_action_batch = torch.zeros_like(input=anchor_action_batch)
        trajectory_action_batch[accepted_steps.anchor_indices] = (
            accepted_steps.predecessor_action_batch
        )
        trajectory_action_batch_per_depth.append(trajectory_action_batch)

        final_error_batch[accepted_steps.anchor_indices] = error_batch[
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

    synthetic_trajectories = _assemble_trajectories(
        accepted_steps_per_depth=accepted_steps_per_depth,
        trajectory_length_batch=trajectory_length_batch,
        state_dimensionality=anchor_state_batch.shape[1],
        action_dimensionality=anchor_action_batch.shape[1],
    )
    return synthetic_trajectories, RolloutDiagnostics(
        anchor_count=anchor_count,
        not_converged_count=not_converged_count,
        final_errors=final_error_batch[trajectory_length_batch > 0],
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
    trajectory_length_batch: torch.Tensor,  # (anchor_count,)
    state_dimensionality: int,
    action_dimensionality: int,
) -> SyntheticTrajectories:
    device = trajectory_length_batch.device

    total_trajectory_length = int(trajectory_length_batch.sum().item())
    states = torch.empty(total_trajectory_length, state_dimensionality, device=device)
    actions = torch.empty(total_trajectory_length, action_dimensionality, device=device)

    anchor_start_index_batch = (
        torch.cumsum(input=trajectory_length_batch, dim=0) - trajectory_length_batch
    )  # (anchor_count,)

    for depth, accepted_steps in enumerate(iterable=accepted_steps_per_depth):
        # The walk found predecessors anchor-first, but a training episode runs the
        # other way, toward the anchor. So the step at depth d lands at the end of
        # its slice, and the deepest one, found last, becomes the episode's start.
        write_index_batch = (
            anchor_start_index_batch[accepted_steps.anchor_indices]
            + trajectory_length_batch[accepted_steps.anchor_indices]
            - 1
            - depth
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

        states[write_index_batch] = accepted_steps.predecessor_state_batch
        actions[write_index_batch] = accepted_steps.predecessor_action_batch

    return SyntheticTrajectories(
        states=states, actions=actions, trajectory_length_batch=trajectory_length_batch
    )
