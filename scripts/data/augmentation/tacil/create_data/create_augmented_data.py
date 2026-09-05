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
from scripts.data.augmentation.tacil.create_data.novelty import (
    measure_distances_to_recorded_states,
)
from scripts.data.augmentation.tacil.create_data.statistics_logging import (
    STAGE_NAME,
    log_dataset_statistics,
    log_rollout_statistics,
)
from scripts.data.augmentation.tacil.create_data.state_distance import (
    calculate_median_step_distance,
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

    action_noise_std = _compute_noise_std(
        samples=episode_store.actions,
        noise_range_fraction=config.action_noise_range_fraction,
    )  # (action_dimensionality,)

    #  We use the median (rather than the mean) because it's robust to outliers.
    median_expert_step_distance = calculate_median_step_distance(
        episode_store=episode_store, dynamics_model_ensemble=dynamics_model_ensemble
    )

    chain_error_limit = (
        config.chain_error_limit_step_multiple * median_expert_step_distance
    )

    backward_trajectories, rollout_diagnostics = roll_out_backward(
        dynamics_model_ensemble=dynamics_model_ensemble,
        anchor_state_batch=torch.as_tensor(episode_store.states[anchor_indices]).to(
            device=device
        ),
        anchor_action_batch=torch.as_tensor(episode_store.actions[anchor_indices]).to(
            device=device
        ),
        action_noise_std=torch.as_tensor(action_noise_std).to(device=device),
        chain_error_criterion=config.chain_error,
        chain_error_limit=chain_error_limit,
        convergence_tolerance=1e-3 * median_expert_step_distance,
    )

    log_rollout_statistics(
        rollout_diagnostics=rollout_diagnostics,
        median_expert_step_distance=median_expert_step_distance,
        chain_error_limit=chain_error_limit,
        run=run,
    )

    if not bool((backward_trajectories.step_count_batch > 0).any()):
        run.summary[f"{STAGE_NAME}/1_rollout/empty_anchor_count"] = len(anchor_indices)
        run.summary[f"{STAGE_NAME}/1_rollout/empty_anchor_fraction"] = 1.0
        raise RuntimeError(
            "The chain error limit rejected every anchor's first backward step, so "
            "this configuration produced no labels. Training on the expert data "
            "alone would score the unaugmented baseline under an augmented "
            "run's name, so the run stops here instead."
        )

    augmentation_result = create_augmented_episodes(
        synthetic_trajectories=backward_trajectories.to_temporal_order(),
        episode_store=episode_store,
        anchor_indices=anchor_indices,
        expert_part_length=expert_part_length,
    )

    log_dataset_statistics(
        augmentation_result=augmentation_result,
        source_episode_count=len(episode_store.episode_end_indices),
        source_transition_count=len(episode_store.states),
        # Only the invented states are asked about: the expert run-out spliced
        # behind each chain is recorded data and its distance is zero by
        # construction.
        distances_to_recorded_states=measure_distances_to_recorded_states(
            synthetic_state_batch=backward_trajectories.predecessor_states,
            recorded_state_batch=torch.as_tensor(data=episode_store.states).to(
                device=device
            ),
            dynamics_model_ensemble=dynamics_model_ensemble,
        )
        / median_expert_step_distance,
        run=run,
    )

    return augmentation_result.episode_store


def _compute_noise_std(
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
