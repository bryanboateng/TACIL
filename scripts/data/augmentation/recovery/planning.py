"""Plans recovery chains from disturbed states back to the demonstrations.

Each chain starts at a disturbed copy of an expert state and should end at
one of its targets: the expert states nearest to where the disturbance left
it. The goal is the demonstrated state distribution, not one particular
state -- once the chain reaches any target, that target's recorded
continuation is what the episode carries on with.

The planner walks one step at a time. At every step it searches for an action
sequence whose predicted end lands near a target, and executes only the first
action of the winner (model-predictive control). The search is a cross-entropy
method: sample candidate sequences, keep the best few, refit the sampling
distribution to them, repeat. Looking several steps ahead is what lets it find
manoeuvres whose first move increases the distance, like walking around the
block before pushing it.

The planner only ever sees the model's imagination, and picking the best of
many candidates is drawn toward states where the model is too optimistic. So
no chain is trusted on the planner's word: every finished chain is replayed
with each ensemble member alone, and only chains whose replay endpoints all
land near the reached target are kept.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from tqdm import tqdm

from scripts.data.augmentation.recovery.config import ActionSource
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
    already_on_manifold_count: int  # disturbance too weak to leave the data
    arrived_count: int  # chains that reached a target
    gave_up_count: int  # chains that hit the length cap before arriving
    replay_rejected_count: int  # arrived, but the members' replay disagreed
    final_replay_errors: torch.Tensor  # (accepted_anchor_count,)


@dataclass(frozen=True)
class RecoveryChains:
    trajectories: SyntheticTrajectories
    # Which of its targets each chain reached, as a column of the target
    # batch. Only meaningful where the chain was accepted.
    reached_target_slot_batch: torch.Tensor  # (anchor_count,)


@torch.no_grad()
def plan_recovery_chains(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    start_state_batch: torch.Tensor,  # (anchor_count, state_dimensionality)
    target_state_batch: torch.Tensor,  # (anchor_count, target_count, state_dimensionality)  # noqa: E501
    action_step_std: torch.Tensor,  # (action_dimensionality,)
    # Called with the active states and which chains they belong to: GPI
    # carries a plan per chain, so it has to know whose turn it is.
    gpi_action_source: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    action_source: ActionSource,
    candidate_count: int,
    search_iteration_count: int,
    elite_fraction: float,
    plan_horizon: int,
    max_chain_length: int,
    success_tolerance: float,
    replay_error_limit: float,
) -> tuple[RecoveryChains, PlanningDiagnostics]:
    anchor_count, state_dimensionality = start_state_batch.shape
    action_dimensionality = action_step_std.shape[0]
    device = start_state_batch.device
    elite_count = max(2, int(candidate_count * elite_fraction))

    # The planner's state runs on the ensemble mean; the members' states are
    # propagated alongside, so the replay test at the end comes for free.
    mean_state_batch = start_state_batch.clone()
    per_member_state_batch = (
        start_state_batch.unsqueeze(dim=0)
        .expand(len(dynamics_model_ensemble.dynamics_models), -1, -1)
        .clone()
    )  # (member_count, anchor_count, state_dimensionality)

    # A start that already sits on the demonstrations has nothing to recover
    # from, and its chain would only teach the policy to stand still.
    start_distance_batch, _ = _distance_to_targets(
        dynamics_model_ensemble=dynamics_model_ensemble,
        state_batch=start_state_batch.unsqueeze(dim=1),
        target_state_batch=target_state_batch,
    )  # (anchor_count, 1)
    already_on_manifold_mask_batch = start_distance_batch.squeeze(dim=1) <= (
        success_tolerance
    )

    active_mask_batch = ~already_on_manifold_mask_batch
    arrived_mask_batch = torch.zeros_like(active_mask_batch)
    step_count_batch = torch.zeros(anchor_count, dtype=torch.long, device=device)
    reached_target_slot_batch = torch.zeros(
        anchor_count, dtype=torch.long, device=device
    )

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
        active_target_batch = target_state_batch[active_indices]

        gpi_action_batch = (
            gpi_action_source(state_batch, active_indices)
            if action_source != "search"
            else None
        )
        if action_source == "gpi":
            action_batch = gpi_action_batch
        else:
            action_batch = _plan_next_action(
                dynamics_model_ensemble=dynamics_model_ensemble,
                state_batch=state_batch,
                target_state_batch=active_target_batch,
                action_step_std=action_step_std,
                prior_action_batch=gpi_action_batch,
                candidate_count=candidate_count,
                search_iteration_count=search_iteration_count,
                elite_count=elite_count,
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

        distance_batch, slot_batch = _distance_to_targets(
            dynamics_model_ensemble=dynamics_model_ensemble,
            state_batch=mean_state_batch[active_indices].unsqueeze(dim=1),
            target_state_batch=active_target_batch,
        )  # (active_count, 1), (active_count, 1)
        arrived_now_mask_batch = distance_batch.squeeze(dim=1) <= success_tolerance
        arrived_indices = active_indices[arrived_now_mask_batch]
        arrived_mask_batch[arrived_indices] = True
        active_mask_batch[arrived_indices] = False
        reached_target_slot_batch[arrived_indices] = slot_batch.squeeze(dim=1)[
            arrived_now_mask_batch
        ]
        progress_bar.update(n=1)
    progress_bar.close()

    # Distilling GPI is not planning: the actions are the label we want to
    # copy, not a proposal that has to prove it works. Requiring arrival would
    # throw away exactly the states where GPI does something interesting, and
    # judging its chains by the model's agreement would measure the model, not
    # GPI. So in that mode every chain is kept and the checks below are only
    # reported.
    verify_chains = action_source != "gpi"

    if verify_chains:
        # Whatever is still active never arrived; those chains teach wandering,
        # not recovery, so they are dropped whole.
        gave_up_count = int(active_mask_batch.sum())
        step_count_batch[active_mask_batch] = 0
    else:
        gave_up_count = 0
        unfinished_indices = active_mask_batch.nonzero().squeeze(dim=1)
        if unfinished_indices.numel() > 0:
            _, slot_batch = _distance_to_targets(
                dynamics_model_ensemble=dynamics_model_ensemble,
                state_batch=mean_state_batch[unfinished_indices].unsqueeze(dim=1),
                target_state_batch=target_state_batch[unfinished_indices],
            )
            # The run-out is spliced from whichever demonstrated state the
            # chain ended up nearest to.
            reached_target_slot_batch[unfinished_indices] = slot_batch.squeeze(dim=1)
            arrived_mask_batch[unfinished_indices] = True
            active_mask_batch[unfinished_indices] = False

    # The independent check: the planner planned on the ensemble mean, the
    # members replayed the same actions alone. Members that end far from the
    # reached target mean the chain leaned on a model error.
    arrived_indices = arrived_mask_batch.nonzero().squeeze(dim=1)
    reached_target_state_batch = target_state_batch[
        arrived_indices, reached_target_slot_batch[arrived_indices]
    ]  # (arrived_count, state_dimensionality)
    replay_error_batch = dynamics_model_ensemble.endpoint_deviation(
        per_member_state_batch=per_member_state_batch[:, arrived_indices],
        target_state_batch=reached_target_state_batch,
    )  # (arrived_count,)
    accepted_mask_batch = (
        replay_error_batch <= replay_error_limit
        if verify_chains
        else torch.ones_like(input=replay_error_batch, dtype=torch.bool)
    )
    step_count_batch[arrived_indices[~accepted_mask_batch]] = 0

    trajectories = _assemble_trajectories(
        chain_state_batch_per_depth=chain_state_batch_per_depth,
        chain_action_batch_per_depth=chain_action_batch_per_depth,
        step_count_batch=step_count_batch,
        state_dimensionality=state_dimensionality,
        action_dimensionality=action_dimensionality,
    )
    return RecoveryChains(
        trajectories=trajectories, reached_target_slot_batch=reached_target_slot_batch
    ), PlanningDiagnostics(
        anchor_count=anchor_count,
        already_on_manifold_count=int(already_on_manifold_mask_batch.sum()),
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
    target_state_batch: torch.Tensor,  # (active_count, target_count, state_dimensionality)  # noqa: E501
    action_step_std: torch.Tensor,  # (action_dimensionality,)
    prior_action_batch: torch.Tensor | None,  # (active_count, action_dimensionality)
    candidate_count: int,
    search_iteration_count: int,
    elite_count: int,
    plan_horizon: int,
) -> torch.Tensor:  # (active_count, action_dimensionality)
    active_count, state_dimensionality = state_batch.shape
    action_dimensionality = action_step_std.shape[0]
    device = state_batch.device
    row_indices = torch.arange(active_count, device=device)

    # Candidate sequences are walks of the agent target, starting at the
    # current agent position. A walk, not independent draws: consecutive
    # expert actions also move in small steps. The search shapes the walk by
    # refitting the step distribution, which starts unbiased and wide.
    step_mean_batch = torch.zeros(
        active_count, plan_horizon, action_dimensionality, device=device
    )
    if prior_action_batch is not None:
        # Start the search at GPI's answer: the first step moves to its target
        # and the rest hold it there, so the sequence the search departs from
        # is one an expert-shaped rule proposed rather than noise.
        step_mean_batch[:, 0] = (
            prior_action_batch - state_batch[:, AGENT_POSITION_SLICE]
        )
    step_std_batch = action_step_std.expand(
        active_count, plan_horizon, action_dimensionality
    ).clone()

    best_score_batch = torch.full((active_count,), float("inf"), device=device)
    best_action_batch = state_batch[:, AGENT_POSITION_SLICE].clone()

    for _ in range(search_iteration_count):
        candidate_step_batch = step_mean_batch[:, None] + step_std_batch[
            :, None
        ] * torch.randn(
            active_count,
            candidate_count,
            plan_horizon,
            action_dimensionality,
            device=device,
        )
        # Candidate 0 is the distribution's mean itself, unperturbed. In the
        # first iteration that is exactly the prior, so a refined chain can
        # only depart from GPI's action when the model prefers something else.
        candidate_step_batch[:, 0] = step_mean_batch

        candidate_action_batch = state_batch[
            :, None, None, AGENT_POSITION_SLICE
        ] + candidate_step_batch.cumsum(dim=2)

        flat_state_batch = (
            state_batch[:, None, :]
            .expand(active_count, candidate_count, state_dimensionality)
            .reshape(-1, state_dimensionality)
        )
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
        # away from the demonstrations can still win.
        score_batch, _ = _distance_to_targets(
            dynamics_model_ensemble=dynamics_model_ensemble,
            state_batch=final_state_batch,
            target_state_batch=target_state_batch,
        )  # (active_count, candidate_count)

        # The winner so far is kept across iterations, so refitting can only
        # add to what the search already found.
        iteration_best_score_batch, iteration_best_index_batch = score_batch.min(dim=1)
        improved_mask_batch = iteration_best_score_batch < best_score_batch
        best_score_batch = torch.where(
            improved_mask_batch, iteration_best_score_batch, best_score_batch
        )
        best_action_batch = torch.where(
            improved_mask_batch[:, None],
            candidate_action_batch[row_indices, iteration_best_index_batch, 0, :],
            best_action_batch,
        )

        elite_index_batch = score_batch.topk(
            k=elite_count, dim=1, largest=False
        ).indices  # (active_count, elite_count)
        elite_step_batch = candidate_step_batch[
            row_indices[:, None], elite_index_batch
        ]  # (active_count, elite_count, plan_horizon, action_dimensionality)
        step_mean_batch = elite_step_batch.mean(dim=1)
        step_std_batch = elite_step_batch.std(dim=1).clamp(min=1e-6)

    return best_action_batch


def _distance_to_targets(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    state_batch: torch.Tensor,  # (active_count, candidate_count, state_dimensionality)
    target_state_batch: torch.Tensor,  # (active_count, target_count, state_dimensionality)  # noqa: E501
) -> tuple[torch.Tensor, torch.Tensor]:  # distances, nearest target slots
    difference_batch = subtract_states(
        minuend_batch=state_batch.unsqueeze(dim=2),
        subtrahend_batch=target_state_batch.unsqueeze(dim=1),
    )  # (active_count, candidate_count, target_count, state_dimensionality)
    distance_batch = torch.linalg.norm(
        dynamics_model_ensemble.state_normalizer.normalize_difference(
            difference_batch=difference_batch
        ),
        dim=-1,
    )  # (active_count, candidate_count, target_count)
    return distance_batch.min(dim=2)


def _assemble_trajectories(
    *,
    chain_state_batch_per_depth: list[torch.Tensor],
    chain_action_batch_per_depth: list[torch.Tensor],
    step_count_batch: torch.Tensor,  # (anchor_count,)
    state_dimensionality: int,
    action_dimensionality: int,
) -> SyntheticTrajectories:
    device = step_count_batch.device

    total_step_count = int(step_count_batch.sum().item())
    states = torch.empty(total_step_count, state_dimensionality, device=device)
    actions = torch.empty(total_step_count, action_dimensionality, device=device)

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
