import numpy as np
import torch
from numpy import typing as npt
from wandb import Run

from scripts.data.augmentation.stitching.config import StitchingDataConfig
from scripts.data.augmentation.stitching.junction_search import (
    Junctions,
    build_episode_index_per_step,
    find_junctions,
)
from scripts.data.augmentation.stitching.statistics_logging import (
    log_dataset_statistics,
    log_junction_statistics,
)
from scripts.data.augmentation.tacil.create_data.step_distance import (
    calculate_median_step_distance,
)
from scripts.data.augmentation.tacil.dynamics.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)
from scripts.shared.episode_store import EpisodeStore
from scripts.shared.seeding import set_seed


def create_stitched_data(
    *,
    config: StitchingDataConfig,
    episode_store: EpisodeStore,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    seed: int,
    run: Run,
) -> EpisodeStore:
    set_seed(seed=seed)

    #  We use the median (rather than the mean) because it's robust to outliers.
    median_expert_step_distance = calculate_median_step_distance(
        episode_store=episode_store, dynamics_model_ensemble=dynamics_model_ensemble
    )
    junction_radius = config.junction_radius_step_multiple * median_expert_step_distance
    junction_error_limit = (
        config.junction_error_limit_step_multiple * median_expert_step_distance
    )

    junctions, junction_diagnostics = find_junctions(
        episode_store=episode_store,
        dynamics_model_ensemble=dynamics_model_ensemble,
        junction_radius=junction_radius,
        junction_error_limit=junction_error_limit,
    )

    log_junction_statistics(
        junctions=junctions,
        junction_diagnostics=junction_diagnostics,
        median_expert_step_distance=median_expert_step_distance,
        junction_radius=junction_radius,
        junction_error_limit=junction_error_limit,
        run=run,
    )

    if len(junctions.source_indices) == 0:
        raise RuntimeError(
            "No pair of steps from different episodes passed the junction "
            "error limit, so this configuration produced no data. Training on "
            "the expert data alone would score the unaugmented baseline under "
            "an augmented run's name, so the run stops here instead. Widen the "
            "search radius or the error limit."
        )

    stitched_episodes, kept_junctions = _assemble_stitched_episodes(
        episode_store=episode_store,
        junctions=junctions,
        transition_budget=int(
            config.transition_budget_multiple * episode_store.states.shape[0]
        ),
    )

    log_dataset_statistics(
        episode_store=episode_store,
        stitched_episodes=stitched_episodes,
        kept_junctions=kept_junctions,
        found_junction_count=len(junctions.source_indices),
        run=run,
    )

    return stitched_episodes


# ---- Private --------------------------------------------


@torch.no_grad()
def _assemble_stitched_episodes(
    *, episode_store: EpisodeStore, junctions: Junctions, transition_budget: int
) -> tuple[EpisodeStore, Junctions]:
    """One new episode per junction: the source's past, the target's future.

    The junction step keeps the source's state and takes the target's action,
    which is the single invented pair in an otherwise recorded episode.
    """
    episode_index_per_step = build_episode_index_per_step(
        episode_end_indices=episode_store.episode_end_indices
    )
    episode_start_indices = np.concatenate(
        (np.zeros(shape=1, dtype=np.int64), episode_store.episode_end_indices[:-1])
    )  # (episode_count,)

    # The smallest error is the most trustworthy splice, so spend the budget
    # there first.
    ranking = np.argsort(a=junctions.errors, kind="stable")

    state_trajectories: list[npt.NDArray[np.float32]] = []
    action_trajectories: list[npt.NDArray[np.float32]] = []
    episode_lengths: list[int] = []
    kept_positions: list[int] = []
    transition_count = 0

    for junction_position in ranking:
        source_index = int(junctions.source_indices[junction_position])
        target_index = int(junctions.target_indices[junction_position])

        source_start_index = int(
            episode_start_indices[episode_index_per_step[source_index]]
        )
        target_end_index = int(
            episode_store.episode_end_indices[episode_index_per_step[target_index]]
        )

        prefix_slice = slice(source_start_index, source_index + 1)
        suffix_slice = slice(target_index + 1, target_end_index)

        episode_length = (prefix_slice.stop - prefix_slice.start) + (
            suffix_slice.stop - suffix_slice.start
        )
        if transition_count + episode_length > transition_budget:
            break

        prefix_actions = episode_store.actions[prefix_slice].copy()
        # The borrowed action is what carries the run into the other episode.
        prefix_actions[-1] = episode_store.actions[target_index]

        state_trajectories.append(
            np.concatenate(
                [
                    episode_store.states[prefix_slice],
                    episode_store.states[suffix_slice],
                ],
                axis=0,
            )
        )
        action_trajectories.append(
            np.concatenate(
                [prefix_actions, episode_store.actions[suffix_slice]], axis=0
            )
        )
        episode_lengths.append(episode_length)
        kept_positions.append(int(junction_position))
        transition_count += episode_length

    if not episode_lengths:
        raise RuntimeError(
            f"The transition budget of {transition_budget} is smaller than the "
            f"shortest stitched episode, so nothing was kept. Raise "
            f"transition_budget_multiple."
        )

    kept_index = np.asarray(a=kept_positions, dtype=np.int64)
    return EpisodeStore(
        states=np.concatenate(state_trajectories, axis=0),
        actions=np.concatenate(action_trajectories, axis=0),
        episode_end_indices=np.cumsum(np.asarray(a=episode_lengths, dtype=np.int64)),
    ), Junctions(
        source_indices=junctions.source_indices[kept_index],
        target_indices=junctions.target_indices[kept_index],
        errors=junctions.errors[kept_index],
    )
