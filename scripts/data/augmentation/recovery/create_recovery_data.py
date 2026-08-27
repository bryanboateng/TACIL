import numpy as np
import torch
from numpy import typing as npt
from wandb import Run

from scripts.data.augmentation.recovery.config import RecoveryDataConfig
from scripts.data.augmentation.recovery.gpi_actions import GpiActionSource
from scripts.data.augmentation.recovery.planning import plan_recovery_chains
from scripts.data.augmentation.recovery.statistics_logging import (
    log_planning_statistics,
)
from scripts.data.augmentation.tacil.create_data.anchor_selection import (
    get_anchor_indices,
)
from scripts.data.augmentation.tacil.create_data.episode_assembly import (
    create_augmented_episodes,
)
from scripts.data.augmentation.tacil.create_data.statistics_logging import (
    STAGE_NAME,
    log_dataset_statistics,
)
from scripts.data.augmentation.tacil.create_data.step_distance import (
    calculate_median_step_distance,
)
from scripts.data.augmentation.tacil.dynamics.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)
from scripts.shared.episode_store import EpisodeStore
from scripts.shared.pusht_state import AGENT_POSITION_SLICE, add_states, subtract_states
from scripts.shared.seeding import set_seed


def create_recovery_data(
    *,
    config: RecoveryDataConfig,
    episode_store: EpisodeStore,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    expert_part_length: int,
    seed: int,
    run: Run,
) -> EpisodeStore:
    set_seed(seed=seed)

    device = next(dynamics_model_ensemble.parameters()).device

    # These are both where the disturbances start and what the chains may aim
    # for: every one of them has a recorded continuation long enough to append.
    anchor_indices = get_anchor_indices(
        episode_end_indices=episode_store.episode_end_indices,
        expert_part_length=expert_part_length,
    )  # (anchor_count,)
    anchor_state_batch = torch.as_tensor(episode_store.states[anchor_indices]).to(
        device=device
    )  # (anchor_count, state_dimensionality)

    #  We use the median (rather than the mean) because it's robust to outliers.
    median_expert_step_distance = calculate_median_step_distance(
        episode_store=episode_store, dynamics_model_ensemble=dynamics_model_ensemble
    )
    replay_error_limit = (
        config.replay_error_limit_step_multiple * median_expert_step_distance
    )
    success_tolerance = (
        config.success_tolerance_step_multiple * median_expert_step_distance
    )

    # A disturbed copy of each anchor: the block is shifted and turned, the
    # agent stays. This is the situation the evaluation disturbances create.
    disturbance_std = _compute_range_fraction_std(
        samples=episode_store.states,
        range_fraction=config.disturbance_range_fraction,
        device=device,
    )  # (state_dimensionality,)
    disturbance_std[AGENT_POSITION_SLICE] = 0.0
    start_state_batch = add_states(
        first_addend_batch=anchor_state_batch,
        second_addend_batch=torch.randn_like(input=anchor_state_batch)
        * disturbance_std,
    )  # (anchor_count, state_dimensionality)

    # Each chain aims for the expert states nearest to where the disturbance
    # left it, not for the anchor it came from: the goal is the demonstrated
    # state distribution, and the nearest states are the ones a short recovery
    # can plausibly reach.
    target_slot_batch = _find_nearest_state_indices(
        query_state_batch=start_state_batch,
        candidate_state_batch=anchor_state_batch,
        dynamics_model_ensemble=dynamics_model_ensemble,
        neighbour_count=min(config.target_count, len(anchor_indices)),
    )  # (anchor_count, target_count)

    action_step_std = _compute_range_fraction_std(
        samples=episode_store.actions,
        range_fraction=config.plan_step_range_fraction,
        device=device,
    )  # (action_dimensionality,)

    recovery_chains, planning_diagnostics = plan_recovery_chains(
        dynamics_model_ensemble=dynamics_model_ensemble,
        start_state_batch=start_state_batch,
        target_state_batch=anchor_state_batch[target_slot_batch],
        action_step_std=action_step_std,
        gpi_action_source=GpiActionSource(
            expert_episode_store=episode_store,
            progression_weight=config.gpi_progression_weight,
            attraction_weight=config.gpi_attraction_weight,
            device=device,
        ),
        action_source=config.action_source,
        candidate_count=config.candidate_count,
        search_iteration_count=config.search_iteration_count,
        elite_fraction=config.elite_fraction,
        plan_horizon=config.plan_horizon,
        max_chain_length=config.max_chain_length,
        success_tolerance=success_tolerance,
        replay_error_limit=replay_error_limit,
    )

    log_planning_statistics(
        planning_diagnostics=planning_diagnostics,
        median_expert_step_distance=median_expert_step_distance,
        replay_error_limit=replay_error_limit,
        run=run,
    )

    if not bool((recovery_chains.trajectories.step_count_batch > 0).any()):
        run.summary[f"{STAGE_NAME}/1_rollout/empty_anchor_count"] = len(anchor_indices)
        run.summary[f"{STAGE_NAME}/1_rollout/empty_anchor_fraction"] = 1.0
        raise RuntimeError(
            "No planned recovery chain reached the demonstrations and passed the "
            "replay check, so this configuration produced no data. Training on "
            "the expert data alone would score the unaugmented baseline under "
            "an augmented run's name, so the run stops here instead."
        )

    # The continuation comes from the state the chain actually reached, which
    # is generally not the anchor the disturbance started from.
    reached_anchor_indices = anchor_indices[
        target_slot_batch[
            torch.arange(len(anchor_indices), device=device),
            recovery_chains.reached_target_slot_batch,
        ]
        .cpu()
        .numpy()
    ]  # (anchor_count,)

    augmentation_result = create_augmented_episodes(
        synthetic_trajectories=recovery_chains.trajectories,
        episode_store=episode_store,
        anchor_indices=reached_anchor_indices,
        expert_part_length=expert_part_length,
    )

    log_dataset_statistics(
        augmentation_result=augmentation_result,
        source_episode_count=len(episode_store.episode_end_indices),
        source_transition_count=len(episode_store.states),
        run=run,
    )

    return augmentation_result.episode_store


# ---- Private --------------------------------------------


def _compute_range_fraction_std(
    *,
    samples: npt.NDArray[np.float32],  # (N, dimensionality)
    range_fraction: float,
    device: torch.device,
) -> torch.Tensor:  # (dimensionality,)
    sample_batch = torch.as_tensor(samples).to(device=device)
    value_range = (
        sample_batch.max(dim=0).values - sample_batch.min(dim=0).values
    )  # (dimensionality,)
    return range_fraction * value_range


@torch.no_grad()
def _find_nearest_state_indices(
    *,
    query_state_batch: torch.Tensor,  # (query_count, state_dimensionality)
    candidate_state_batch: torch.Tensor,  # (candidate_count, state_dimensionality)
    dynamics_model_ensemble: DynamicsModelEnsemble,
    neighbour_count: int,
) -> torch.Tensor:  # (query_count, neighbour_count)
    nearest_index_batches: list[torch.Tensor] = []
    chunk_size = 256
    for start in range(0, query_state_batch.shape[0], chunk_size):
        query_chunk = query_state_batch[start : start + chunk_size]
        distance_batch = torch.linalg.norm(
            dynamics_model_ensemble.state_normalizer.normalize_difference(
                difference_batch=subtract_states(
                    minuend_batch=query_chunk.unsqueeze(dim=1),
                    subtrahend_batch=candidate_state_batch.unsqueeze(dim=0),
                )
            ),
            dim=-1,
        )  # (chunk_size, candidate_count)
        nearest_index_batches.append(
            distance_batch.topk(k=neighbour_count, dim=1, largest=False).indices
        )
    return torch.cat(tensors=nearest_index_batches)
