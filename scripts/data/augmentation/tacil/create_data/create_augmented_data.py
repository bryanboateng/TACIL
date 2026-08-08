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
from scripts.data.augmentation.tacil.create_data.statistics_logging import (
    STAGE_PREFIX,
    log_dataset_statistics,
    log_rollout_statistics,
)
from scripts.data.augmentation.tacil.create_data.step_distance import (
    calculate_median_step_distance,
)
from scripts.data.augmentation.tacil.models.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)
from scripts.data.base.episode_store import EpisodeStore
from scripts.shared.log import create_logger
from scripts.shared.seeding import set_seed

logger = create_logger()


def create_augmented_data(
    *,
    config: DataConfig,
    episode_store: EpisodeStore,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    prediction_horizon: int | None,
    seed: int,
    device: torch.device,
    run: Run,
) -> EpisodeStore | None:
    """Roll backward out of anchors picked from *episode_store* and return the
    synthetic episodes alone, logging statistics to *run*. None means the
    rollout produced nothing.

    *prediction_horizon* is the window length the consumer trains on, and None
    means it trains on single (state, action) pairs. A windowed consumer needs
    enough expert continuation behind every synthetic state to fill one window;
    the other kind needs none, and copying expert data would only drown its
    labels.
    """
    set_seed(seed=seed)

    expert_part_length = prediction_horizon - 1 if prediction_horizon is not None else 0

    anchor_indices = get_anchor_indices(
        episode_end_indices=episode_store.episode_end_indices,
        expert_part_length=expert_part_length,
    )  # (anchor_count,)

    action_noise_std = compute_noise_std(
        samples=episode_store.actions, noise_fraction=config.action_noise_fraction
    )  # (action_dimensionality,)

    #  The median expert step distance is useful as a reference scale.
    #  It answers the question "How far does the expert normally move in one timestep?"
    #  It can be used to judge whether some other quantity (like a predicted or generated step)
    #  is unusually large or small.
    #
    #  We use the median (rather than the mean) because it's robust to
    #  outliers.
    #  A handful of unusually large jumps won't skew the result the way they would an average.
    median_expert_step_distance = calculate_median_step_distance(
        episode_store=episode_store, dynamics_model_ensemble=dynamics_model_ensemble
    )

    error_limit = config.error_limit_fraction * median_expert_step_distance

    backward_trajectories, rollout_diagnostics = roll_out_backward(
        dynamics_model_ensemble=dynamics_model_ensemble,
        anchor_state_batch=torch.as_tensor(episode_store.states[anchor_indices]).to(
            device=device
        ),
        anchor_action_batch=torch.as_tensor(episode_store.actions[anchor_indices]).to(
            device=device
        ),
        action_noise_std=torch.as_tensor(action_noise_std).to(device=device),
        disagreement_weight=config.disagreement_weight,
        error_limit=error_limit,
        convergence_tolerance=1e-3 * median_expert_step_distance,
    )

    log_rollout_statistics(
        rollout_diagnostics=rollout_diagnostics,
        median_expert_step_distance=median_expert_step_distance,
        error_limit=error_limit,
        run=run,
    )

    if not bool((backward_trajectories.step_count_batch > 0).any()):
        # Not a failure: the budget rejected every first backward step, so
        # this configuration contributes no labels and the policy trains on
        # the expert data alone. Reporting it as an ordinary trial lets the
        # sweep see the reward such a setting earns; raising would leave the
        # whole corner a blind spot it keeps proposing into — and enough
        # failures in a row make the agent give up altogether.
        logger.warning(
            "No anchor survived its first backward step, so no labels were "
            "generated. Continuing with the expert data alone."
        )
        run.summary[f"{STAGE_PREFIX}2_rollout/accepted_anchor_count"] = 0
        run.summary[f"{STAGE_PREFIX}2_rollout/accepted_anchor_fraction"] = 0.0
        return None

    augmentation_result = create_augmented_episodes(
        backward_trajectories=backward_trajectories,
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


def compute_noise_std(
    *,
    samples: npt.NDArray[np.float32],  # (N, dimensionality)
    noise_fraction: float,
) -> npt.NDArray[np.float32]:  # (dimensionality,)
    min_per_dimension: npt.NDArray[np.float32] = samples.min(
        axis=0
    )  # (dimensionality,)
    max_per_dimension: npt.NDArray[np.float32] = samples.max(
        axis=0
    )  # (dimensionality,)
    range_per_dimension = max_per_dimension - min_per_dimension  # (dimensionality,)
    return noise_fraction * range_per_dimension
