import numpy as np
import torch
from numpy import typing as npt
from wandb import Run

from scripts.data.augmentation.recovery.config import RecoveryDataConfig
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
from scripts.shared.pusht_state import AGENT_POSITION_SLICE, add_states
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

    action_step_std = _compute_range_fraction_std(
        samples=episode_store.actions,
        range_fraction=config.plan_step_range_fraction,
        device=device,
    )  # (action_dimensionality,)

    synthetic_trajectories, planning_diagnostics = plan_recovery_chains(
        dynamics_model_ensemble=dynamics_model_ensemble,
        start_state_batch=start_state_batch,
        anchor_state_batch=anchor_state_batch,
        action_step_std=action_step_std,
        candidate_count=config.candidate_count,
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

    if not bool((synthetic_trajectories.step_count_batch > 0).any()):
        run.summary[f"{STAGE_NAME}/1_rollout/empty_anchor_count"] = len(anchor_indices)
        run.summary[f"{STAGE_NAME}/1_rollout/empty_anchor_fraction"] = 1.0
        raise RuntimeError(
            "No planned recovery chain arrived at its anchor and passed the "
            "replay check, so this configuration produced no data. Training on "
            "the expert data alone would score the unaugmented baseline under "
            "an augmented run's name, so the run stops here instead."
        )

    augmentation_result = create_augmented_episodes(
        synthetic_trajectories=synthetic_trajectories,
        episode_store=episode_store,
        anchor_indices=anchor_indices,
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
