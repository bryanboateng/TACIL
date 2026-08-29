import numpy as np
import torch
from numpy import typing as npt
from wandb import Run

from scripts.data.augmentation.distillation.config import DistillationDataConfig
from scripts.data.augmentation.distillation.gpi_actions import GpiActionSource
from scripts.data.augmentation.distillation.rollout import follow_gpi
from scripts.data.augmentation.distillation.statistics_logging import (
    log_rollout_statistics,
)
from scripts.data.augmentation.tacil.create_data.anchor_selection import (
    get_anchor_indices,
)
from scripts.data.augmentation.tacil.create_data.episode_assembly import (
    create_augmented_episodes,
)
from scripts.data.augmentation.tacil.create_data.statistics_logging import (
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


def create_distillation_data(
    *,
    config: DistillationDataConfig,
    episode_store: EpisodeStore,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    expert_part_length: int,
    seed: int,
    run: Run,
) -> EpisodeStore:
    set_seed(seed=seed)

    device = next(dynamics_model_ensemble.parameters()).device

    # These are both where the disturbances start and what a chain may return
    # to: every one of them has a recorded continuation long enough to append.
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
    on_manifold_tolerance = (
        config.on_manifold_tolerance_step_multiple * median_expert_step_distance
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

    distilled_chains, rollout_diagnostics = follow_gpi(
        dynamics_model_ensemble=dynamics_model_ensemble,
        start_state_batch=start_state_batch,
        demonstration_state_batch=anchor_state_batch,
        gpi_action_source=GpiActionSource(
            expert_episode_store=episode_store,
            progression_weight=config.gpi_progression_weight,
            attraction_weight=config.gpi_attraction_weight,
            plan_length=config.gpi_plan_length,
            chain_count=len(anchor_indices),
            device=device,
        ),
        max_chain_length=config.max_chain_length,
        on_manifold_tolerance=on_manifold_tolerance,
    )

    log_rollout_statistics(
        rollout_diagnostics=rollout_diagnostics,
        median_expert_step_distance=median_expert_step_distance,
        on_manifold_tolerance=on_manifold_tolerance,
        run=run,
    )

    if not bool((distilled_chains.trajectories.step_count_batch > 0).any()):
        raise RuntimeError(
            "Every disturbed state already sat on the demonstrations, so no "
            "chain was followed and this configuration produced no data. "
            "Training on the expert data alone would score the unaugmented "
            "baseline under an augmented run's name, so the run stops here "
            "instead. Raise the disturbance, or lower the tolerance."
        )

    augmentation_result = create_augmented_episodes(
        synthetic_trajectories=distilled_chains.trajectories,
        episode_store=episode_store,
        # The continuation comes from the recorded state the chain ended up
        # nearest to, which is generally not the anchor it started from.
        anchor_indices=anchor_indices[
            distilled_chains.nearest_demonstration_index_batch.cpu().numpy()
        ],
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
