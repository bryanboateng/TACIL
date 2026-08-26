"""Plans recovery chains from disturbed states back to their anchors.

Each chain starts at a disturbed copy of an expert state and should end at
that expert state. The planner walks one step at a time: it samples many
candidate action sequences, rolls each out with the ensemble-mean dynamics
model, and executes the first action of the sequence whose predicted end
lands closest to the anchor (random-shooting MPC). Looking a few steps ahead
is what lets it find manoeuvres whose first move increases the distance,
like walking around the block before pushing it back.

The planner only ever sees the model's imagination, and picking the best of
many candidates is drawn toward states where the model is too optimistic. So
no chain is trusted on the planner's word: every finished chain is replayed
with each ensemble member alone, and only chains whose replay endpoints all
land near the anchor are kept.
"""

from dataclasses import dataclass

import torch
from tqdm import tqdm

from scripts.data.augmentation.tacil.create_data.episode_assembly import (
    SyntheticTrajectories,
)
from scripts.data.augmentation.tacil.dynamics.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)
from scripts.shared.pusht_state import AGENT_POSITION_SLICE, subtract_states


@dataclass(frozen=True)
class PlanningDiagnostics:
    anchor_count: int
    arrived_count: int  # chains that reached the success tolerance
    gave_up_count: int  # chains that hit the length cap before arriving
    replay_rejected_count: int  # arrived, but the members' replay disagreed
    final_replay_errors: torch.Tensor  # (accepted_anchor_count,)


@torch.no_grad()
def plan_recovery_chains(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    start_state_batch: torch.Tensor,  # (anchor_count, state_dimensionality)
    anchor_state_batch: torch.Tensor,  # (anchor_count, state_dimensionality)
    action_step_std: torch.Tensor,  # (action_dimensionality,)
    candidate_count: int,
    plan_horizon: int,
    max_chain_length: int,
    success_tolerance: float,
    replay_error_limit: float,
) -> tuple[SyntheticTrajectories, PlanningDiagnostics]:
    anchor_count, state_dimensionality = start_state_batch.shape
    action_dimensionality = action_step_std.shape[0]
    device = start_state_batch.device

    # The planner's state runs on the ensemble mean; the members' states are
    # propagated alongside, so the replay test at the end comes for free.
    mean_state_batch = start_state_batch.clone()
    per_member_state_batch = (
        start_state_batch.unsqueeze(dim=0)
        .expand(len(dynamics_model_ensemble.dynamics_models), -1, -1)
        .clone()
    )  # (member_count, anchor_count, state_dimensionality)

    active_mask_batch = torch.ones(anchor_count, dtype=torch.bool, device=device)
    arrived_mask_batch = torch.zeros_like(active_mask_batch)
    step_count_batch = torch.zeros(anchor_count, dtype=torch.long, device=device)

    # Element d holds the (state, action) pairs recorded at depth d; anchors
    # that stopped earlier keep zeros there, which are never read.
    chain_state_batch_per_depth: list[torch.Tensor] = []
    chain_action_batch_per_depth: list[torch.Tensor] = []

    progress_bar = tqdm(total=max_chain_length, desc="recovery step", unit="step")
    for depth in range(max_chain_length):
        active_indices = active_mask_batch.nonzero().squeeze(dim=1)  # (active_count,)
        if active_indices.numel() == 0:
            break
        progress_bar.set_postfix(active_anchor_count=int(active_indices.numel()))

        state_batch = mean_state_batch[active_indices]
        target_batch = anchor_state_batch[active_indices]

        action_batch = _plan_next_action(
            dynamics_model_ensemble=dynamics_model_ensemble,
            state_batch=state_batch,
            target_batch=target_batch,
            action_step_std=action_step_std,
            candidate_count=candidate_count,
            plan_horizon=plan_horizon,
        )  # (active_count, action_dimensionality)

        # Record the pair before stepping: this action is what the policy
        # should output when it sees this state.
        depth_state_batch = torch.zeros(
            anchor_count, state_dimensionality, device=device
        )
        depth_state_batch[active_indices] = state_batch
        chain_state_batch_per_depth.append(depth_state_batch)
        depth_action_batch = torch.zeros(
            anchor_count, action_dimensionality, device=device
        )
        depth_action_batch[active_indices] = action_batch
        chain_action_batch_per_depth.append(depth_action_batch)
        step_count_batch[active_indices] = depth + 1

        mean_state_batch[active_indices] = dynamics_model_ensemble.predict_next_state(
            state_batch=state_batch, action_batch=action_batch
        )
        for member_index, dynamics_model in enumerate(
            iterable=dynamics_model_ensemble.dynamics_models
        ):
            per_member_state_batch[member_index, active_indices] = (
                dynamics_model.predict_next_state(
                    state_batch=per_member_state_batch[member_index, active_indices],
                    action_batch=action_batch,
                )
            )

        distance_batch = _distance_to_target(
            dynamics_model_ensemble=dynamics_model_ensemble,
            state_batch=mean_state_batch[active_indices],
            target_batch=target_batch,
        )  # (active_count,)
        arrived_now_mask_batch = distance_batch <= success_tolerance
        arrived_mask_batch[active_indices[arrived_now_mask_batch]] = True
        active_mask_batch[active_indices[arrived_now_mask_batch]] = False
        progress_bar.update(n=1)
    progress_bar.close()

    # Whatever is still active never arrived; those chains teach wandering,
    # not recovery, so they are dropped whole.
    gave_up_count = int(active_mask_batch.sum())
    step_count_batch[active_mask_batch] = 0

    # The independent check: the planner planned on the ensemble mean, the
    # members replayed the same actions alone. Members that end far from the
    # anchor mean the chain leaned on a model error.
    arrived_indices = arrived_mask_batch.nonzero().squeeze(dim=1)
    replay_error_batch = dynamics_model_ensemble.endpoint_deviation(
        per_member_state_batch=per_member_state_batch[:, arrived_indices],
        target_state_batch=anchor_state_batch[arrived_indices],
    )  # (arrived_count,)
    accepted_mask_batch = replay_error_batch <= replay_error_limit
    step_count_batch[arrived_indices[~accepted_mask_batch]] = 0

    synthetic_trajectories = _assemble_trajectories(
        chain_state_batch_per_depth=chain_state_batch_per_depth,
        chain_action_batch_per_depth=chain_action_batch_per_depth,
        step_count_batch=step_count_batch,
    )
    return synthetic_trajectories, PlanningDiagnostics(
        anchor_count=anchor_count,
        arrived_count=int(arrived_mask_batch.sum()),
        gave_up_count=gave_up_count,
        replay_rejected_count=int((~accepted_mask_batch).sum()),
        final_replay_errors=replay_error_batch[accepted_mask_batch],
    )


# ---- Private --------------------------------------------


def _plan_next_action(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    state_batch: torch.Tensor,  # (active_count, state_dimensionality)
    target_batch: torch.Tensor,  # (active_count, state_dimensionality)
    action_step_std: torch.Tensor,  # (action_dimensionality,)
    candidate_count: int,
    plan_horizon: int,
) -> torch.Tensor:  # (active_count, action_dimensionality)
    active_count, state_dimensionality = state_batch.shape
    action_dimensionality = action_step_std.shape[0]
    device = state_batch.device

    # Candidate sequences are random walks of the agent target, starting at
    # the current agent position. A walk, not independent draws: consecutive
    # expert actions also move in small steps.
    candidate_step_batch = (
        torch.randn(
            active_count,
            candidate_count,
            plan_horizon,
            action_dimensionality,
            device=device,
        )
        * action_step_std
    )
    candidate_action_batch = state_batch[
        :, None, None, AGENT_POSITION_SLICE
    ] + candidate_step_batch.cumsum(
        dim=2
    )  # (active_count, candidate_count, plan_horizon, action_dimensionality)

    flat_state_batch = (
        state_batch[:, None, :]
        .expand(active_count, candidate_count, state_dimensionality)
        .reshape(-1, state_dimensionality)
    )  # (active_count * candidate_count, state_dimensionality)
    for horizon_index in range(plan_horizon):
        flat_state_batch = dynamics_model_ensemble.predict_next_state(
            state_batch=flat_state_batch,
            action_batch=candidate_action_batch[:, :, horizon_index, :].reshape(
                -1, action_dimensionality
            ),
        )
    final_state_batch = flat_state_batch.view(
        active_count, candidate_count, state_dimensionality
    )

    # Judged on where the whole sequence ends, so a first step that moves
    # away from the anchor can still win.
    score_batch = _distance_to_target(
        dynamics_model_ensemble=dynamics_model_ensemble,
        state_batch=final_state_batch,
        target_batch=target_batch[:, None, :],
    )  # (active_count, candidate_count)
    best_candidate_batch = score_batch.argmin(dim=1)  # (active_count,)

    # Only the first action is executed; the next chain step replans.
    return candidate_action_batch[
        torch.arange(active_count, device=device), best_candidate_batch, 0, :
    ]


def _distance_to_target(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    state_batch: torch.Tensor,  # (..., state_dimensionality)
    target_batch: torch.Tensor,  # (..., state_dimensionality), broadcastable
) -> torch.Tensor:  # (...)
    return torch.linalg.norm(
        dynamics_model_ensemble.state_normalizer.normalize_difference(
            difference_batch=subtract_states(
                minuend_batch=state_batch, subtrahend_batch=target_batch
            )
        ),
        dim=-1,
    )


def _assemble_trajectories(
    *,
    chain_state_batch_per_depth: list[torch.Tensor],
    chain_action_batch_per_depth: list[torch.Tensor],
    step_count_batch: torch.Tensor,  # (anchor_count,)
) -> SyntheticTrajectories:
    device = step_count_batch.device

    total_step_count = int(step_count_batch.sum().item())
    states = torch.empty(
        total_step_count, chain_state_batch_per_depth[0].shape[1], device=device
    )
    actions = torch.empty(
        total_step_count, chain_action_batch_per_depth[0].shape[1], device=device
    )

    start_index_batch = (
        torch.cumsum(input=step_count_batch, dim=0) - step_count_batch
    )  # (anchor_count,)

    for depth, (depth_state_batch, depth_action_batch) in enumerate(
        iterable=zip(chain_state_batch_per_depth, chain_action_batch_per_depth)
    ):
        chain_indices = (step_count_batch > depth).nonzero().squeeze(dim=1)
        if chain_indices.numel() == 0:
            continue
        write_index_batch = start_index_batch[chain_indices] + depth
        states[write_index_batch] = depth_state_batch[chain_indices]
        actions[write_index_batch] = depth_action_batch[chain_indices]

    # The chains were planned forward, so they are already in temporal order.
    return SyntheticTrajectories(
        states=states, actions=actions, step_count_batch=step_count_batch
    )
