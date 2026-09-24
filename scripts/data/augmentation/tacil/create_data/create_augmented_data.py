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
from scripts.data.augmentation.tacil.create_data.distances import (
    calculate_median_action_step_distance,
    calculate_median_state_step_distance,
)
from scripts.data.augmentation.tacil.create_data.episode_assembly import (
    create_augmented_episodes,
)
from scripts.data.augmentation.tacil.create_data.nearest_expert_state import (
    calculate_distances_to_nearest_expert_state,
    calculate_distances_within_expert_states,
)
from scripts.data.augmentation.tacil.create_data.statistics_logging import (
    STAGE_NAME,
    log_dataset_statistics,
    log_median_expert_step_distances,
    log_rollout_statistics,
)
from scripts.data.augmentation.tacil.dynamics.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)
from scripts.shared.episodes import Episodes
from scripts.shared.seeding import set_seed


def create_augmented_data(
    *,
    config: DataConfig,
    expert_episodes: Episodes,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    expert_part_length: int,
    seed: int,
    run: Run,
) -> Episodes:
    set_seed(seed=seed)

    device = next(dynamics_model_ensemble.parameters()).device

    anchor_indices = get_anchor_indices(
        episode_end_indices=expert_episodes.episode_end_indices,
        expert_part_length=expert_part_length,
    )  # (anchor_count,)

    action_noise_std = _calculate_noise_std(
        samples=expert_episodes.actions, noise_range_fraction=config.action_noise
    )  # (action_dimensionality,)

    median_expert_state_step_distance = calculate_median_state_step_distance(
        episodes=expert_episodes, dynamics_model_ensemble=dynamics_model_ensemble
    )
    median_expert_action_step_distance = calculate_median_action_step_distance(
        episodes=expert_episodes
    )
    log_median_expert_step_distances(
        median_expert_state_step_distance=median_expert_state_step_distance,
        median_expert_action_step_distance=median_expert_action_step_distance,
        run=run,
    )

    error_limit = config.error_limit_in_steps * median_expert_state_step_distance

    synthetic_trajectories, rollout_diagnostics = roll_out_backward(
        dynamics_model_ensemble=dynamics_model_ensemble,
        anchor_state_batch=torch.as_tensor(expert_episodes.states[anchor_indices]).to(
            device=device
        ),
        anchor_action_batch=torch.as_tensor(expert_episodes.actions[anchor_indices]).to(
            device=device
        ),
        action_noise_std=torch.as_tensor(action_noise_std).to(device=device),
        stopping_rule=config.stopping_rule,
        error_limit=error_limit,
        convergence_tolerance=1e-3 * median_expert_state_step_distance,
    )

    log_rollout_statistics(
        rollout_diagnostics=rollout_diagnostics, error_limit=error_limit, run=run
    )

    if not bool((synthetic_trajectories.trajectory_lengths > 0).any()):
        run.summary[f"{STAGE_NAME}/2_rollout/4_empty_anchor_count"] = len(
            anchor_indices
        )
        run.summary[f"{STAGE_NAME}/2_rollout/5_empty_anchor_fraction"] = 1.0
        raise RuntimeError(
            "The trajectory error limit rejected every anchor's first backward step, "
            "so no synthetic transitions were produced."
        )

    augmentation_result = create_augmented_episodes(
        synthetic_trajectories=synthetic_trajectories,
        expert_episodes=expert_episodes,
        anchor_indices=anchor_indices,
        expert_part_length=expert_part_length,
    )

    expert_state_batch = torch.as_tensor(data=expert_episodes.states).to(
        device=device
    )  # (expert_count, state_dimensionality)
    expert_action_batch = torch.as_tensor(data=expert_episodes.actions).to(
        device=device
    )  # (expert_count, action_dimensionality)

    log_dataset_statistics(
        augmentation_result=augmentation_result,
        source_episode_count=len(expert_episodes.episode_end_indices),
        source_transition_count=len(expert_episodes.states),
        synthetic_distances=calculate_distances_to_nearest_expert_state(
            query_state_batch=synthetic_trajectories.states,
            query_action_batch=synthetic_trajectories.actions,
            expert_state_batch=expert_state_batch,
            expert_action_batch=expert_action_batch,
            dynamics_model_ensemble=dynamics_model_ensemble,
        ),
        expert_distances=calculate_distances_within_expert_states(
            expert_state_batch=expert_state_batch,
            expert_action_batch=expert_action_batch,
            dynamics_model_ensemble=dynamics_model_ensemble,
        ),
        median_expert_state_step_distance=median_expert_state_step_distance,
        median_expert_action_step_distance=median_expert_action_step_distance,
        run=run,
    )

    return augmentation_result.episodes


# ---- Private --------------------------------------------


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
