import numpy as np
import torch
from numpy import typing as npt
from wandb import Run

from scripts.data.augmentation.tacil.create_data.anchor_selection import (
    get_anchor_indices,
)
from scripts.data.augmentation.tacil.create_data.backward_rollout import (
    roll_out_backward,
)
from scripts.data.augmentation.tacil.create_data.config import DataConfig
from scripts.data.augmentation.tacil.create_data.episode_assembly import (
    create_augmented_episodes,
)
from scripts.data.augmentation.tacil.create_data.nearest_recorded_state import (
    calculate_distances_to_nearest_recorded_state,
)
from scripts.data.augmentation.tacil.create_data.state_distance import (
    calculate_step_distance,
)
from scripts.data.augmentation.tacil.create_data.statistics_logging import (
    STAGE_NAME,
    log_dataset_statistics,
    log_rollout_statistics,
)
from scripts.data.augmentation.tacil.dynamics.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)
from scripts.shared.episode_store import EpisodeStore
from scripts.shared.seeding import set_seed


def create_augmented_data(
    *,
    config: DataConfig,
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

    action_noise_std = _calculate_noise_std(
        samples=episode_store.actions, noise_range_fraction=config.action_noise
    )  # (action_dimensionality,)

    #  We use the median (rather than the mean) because it's robust to outliers.
    expert_step_distance = calculate_step_distance(
        episode_store=episode_store, dynamics_model_ensemble=dynamics_model_ensemble
    )

    error_limit = config.error_limit_in_steps * expert_step_distance

    synthetic_trajectories, rollout_diagnostics = roll_out_backward(
        dynamics_model_ensemble=dynamics_model_ensemble,
        anchor_state_batch=torch.as_tensor(episode_store.states[anchor_indices]).to(
            device=device
        ),
        anchor_action_batch=torch.as_tensor(episode_store.actions[anchor_indices]).to(
            device=device
        ),
        action_noise_std=torch.as_tensor(action_noise_std).to(device=device),
        stopping_rule=config.stopping_rule,
        error_limit=error_limit,
        convergence_tolerance=1e-3 * expert_step_distance,
    )

    log_rollout_statistics(
        rollout_diagnostics=rollout_diagnostics,
        expert_step_distance=expert_step_distance,
        error_limit=error_limit,
        run=run,
    )

    if not bool((synthetic_trajectories.trajectory_lengths > 0).any()):
        run.summary[f"{STAGE_NAME}/1_rollout/5_empty_anchor_count"] = len(
            anchor_indices
        )
        run.summary[f"{STAGE_NAME}/1_rollout/6_empty_anchor_fraction"] = 1.0
        raise RuntimeError(
            "The trajectory error limit rejected every anchor's first backward step, so "
            "this configuration produced no labels. Training on the expert data "
            "alone would score the unaugmented baseline under an augmented "
            "run's name, so the run stops here instead."
        )

    augmentation_result = create_augmented_episodes(
        synthetic_trajectories=synthetic_trajectories,
        expert_episodes=episode_store,
        anchor_indices=anchor_indices,
        expert_part_length=expert_part_length,
    )

    recorded_state_batch = torch.as_tensor(data=episode_store.states).to(
        device=device
    )  # (recorded_count, state_dimensionality)
    recorded_action_batch = torch.as_tensor(data=episode_store.actions).to(
        device=device
    )  # (recorded_count, action_dimensionality)

    log_dataset_statistics(
        augmentation_result=augmentation_result,
        source_episode_count=len(episode_store.episode_end_indices),
        source_transition_count=len(episode_store.states),
        # Only the invented steps are asked about: the expert run-out spliced
        # behind each trajectory is recorded data and its distance is zero by
        # construction.
        synthetic_distances=calculate_distances_to_nearest_recorded_state(
            query_state_batch=synthetic_trajectories.states,
            query_action_batch=synthetic_trajectories.actions,
            recorded_state_batch=recorded_state_batch,
            recorded_action_batch=recorded_action_batch,
            dynamics_model_ensemble=dynamics_model_ensemble,
            queries_are_the_recorded_states=False,
        ),
        # The same measurement inside the recorded data, which is what says
        # whether the numbers above are unusual.
        recorded_distances=calculate_distances_to_nearest_recorded_state(
            query_state_batch=recorded_state_batch,
            query_action_batch=recorded_action_batch,
            recorded_state_batch=recorded_state_batch,
            recorded_action_batch=recorded_action_batch,
            dynamics_model_ensemble=dynamics_model_ensemble,
            queries_are_the_recorded_states=True,
        ),
        expert_step_distance=expert_step_distance,
        run=run,
    )

    return augmentation_result.episode_store


def _calculate_noise_std(
    *,
    samples: npt.NDArray[np.float32],  # (N, dimensionality)
    noise_range_fraction: float,
) -> npt.NDArray[np.float32]:  # (dimensionality,)
    min_per_dimension: npt.NDArray[np.float32] = samples.min(
        axis=0
    )  # (dimensionality,)
    max_per_dimension: npt.NDArray[np.float32] = samples.max(
        axis=0
    )  # (dimensionality,)
    range_per_dimension = max_per_dimension - min_per_dimension  # (dimensionality,)
    return noise_range_fraction * range_per_dimension
