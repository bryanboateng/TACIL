from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch

from scripts.data.augmentation.tacil.create_data.backward_rollout import (
    BackwardTrajectories,
)
from scripts.shared.data import EpisodeStore


@dataclass(frozen=True)
class AugmentationResult:
    @dataclass(frozen=True)
    class OutcomeCounts:
        empty: int  # anchors whose first backward step already broke the budget
        accepted: int

    episode_store: EpisodeStore
    synthetic_part_lengths: npt.NDArray[np.int64]  # (accepted_count,)
    counts: OutcomeCounts


@torch.no_grad()
def create_augmented_episodes(
    *,
    backward_trajectories: BackwardTrajectories,
    episode_store: EpisodeStore,
    anchor_indices: npt.NDArray[np.int64],  # (anchor_count,)
    expert_part_length: int,
) -> AugmentationResult:
    synthetic_part_lengths: npt.NDArray[np.int64] = (
        backward_trajectories.step_count_batch.cpu().numpy()
    )  # (anchor_count,)
    accepted_anchor_count = int((synthetic_part_lengths > 0).sum())
    empty_anchor_count = int(synthetic_part_lengths.shape[0]) - accepted_anchor_count
    if accepted_anchor_count == 0:
        raise RuntimeError(
            "The error budget produced no augmented episodes. Raise "
            "error_budget_fraction or retrain the dynamics model."
        )

    augmented_episode_store = _assemble_episodes(
        expert_states=episode_store.states,
        expert_actions=episode_store.actions,
        synthetic_state_trajectory_batch=backward_trajectories.predecessor_states_batch.cpu().numpy(),
        synthetic_action_trajectory_batch=backward_trajectories.predecessor_actions_batch.cpu().numpy(),
        synthetic_part_length_batch=synthetic_part_lengths,
        anchor_index_batch=np.asarray(anchor_indices, dtype=np.int64),
        expert_part_length=expert_part_length,
    )

    return AugmentationResult(
        episode_store=augmented_episode_store,
        synthetic_part_lengths=synthetic_part_lengths[synthetic_part_lengths > 0],
        counts=AugmentationResult.OutcomeCounts(
            empty=empty_anchor_count, accepted=accepted_anchor_count
        ),
    )


def _assemble_episodes(
    *,
    expert_states: npt.NDArray[
        np.float32
    ],  # (total_transition_count, state_dimensionality)
    expert_actions: npt.NDArray[
        np.float32
    ],  # (total_transition_count, action_dimensionality)
    synthetic_state_trajectory_batch: npt.NDArray[
        np.float32
    ],  # (anchor_count, step_count, state_dimensionality)
    synthetic_action_trajectory_batch: npt.NDArray[
        np.float32
    ],  # (anchor_count, step_count, action_dimensionality)
    synthetic_part_length_batch: npt.NDArray[np.int64],  # (anchor_count,)
    anchor_index_batch: npt.NDArray[np.int64],  # (anchor_count,)
    expert_part_length: int,
) -> EpisodeStore:
    accepted_anchor_mask = synthetic_part_length_batch > 0  # (anchor_count,)

    state_trajectories_per_episode: list[np.ndarray] = []
    action_trajectories_per_episode: list[np.ndarray] = []
    episode_lengths: list[int] = []

    for (
        synthetic_part_length,
        anchor_index,
        synthetic_state_trajectory,
        synthetic_action_trajectory,
    ) in zip(
        synthetic_part_length_batch[accepted_anchor_mask],
        anchor_index_batch[accepted_anchor_mask],
        synthetic_state_trajectory_batch[accepted_anchor_mask],
        synthetic_action_trajectory_batch[accepted_anchor_mask],
    ):
        # Synthetic prefix in temporal order = predecessors reversed.
        time_ascending_synthetic_state_trajectory = synthetic_state_trajectory[
            :synthetic_part_length
        ][::-1]  # (synthetic_part_length, state_dimensionality)
        time_ascending_synthetic_action_trajectory = synthetic_action_trajectory[
            :synthetic_part_length
        ][::-1]  # (synthetic_part_length, action_dimensionality)

        expert_part_slice = slice(anchor_index, anchor_index + expert_part_length)

        state_trajectories_per_episode.append(
            np.concatenate(
                [
                    time_ascending_synthetic_state_trajectory,
                    expert_states[expert_part_slice],
                ],
                axis=0,
            )
        )
        action_trajectories_per_episode.append(
            np.concatenate(
                [
                    time_ascending_synthetic_action_trajectory,
                    expert_actions[expert_part_slice],
                ],
                axis=0,
            )
        )
        episode_lengths.append(int(synthetic_part_length) + expert_part_length)

    return EpisodeStore(
        states=np.concatenate(
            state_trajectories_per_episode, axis=0
        ),  # (total_timestep_count, state_dimensionality)
        actions=np.concatenate(
            action_trajectories_per_episode, axis=0
        ),  # (total_timestep_count, action_dimensionality)
        episode_end_indices=np.cumsum(
            np.asarray(episode_lengths, dtype=np.int64)
        ),  # (episode_count,)
        # These episodes end mid-task at an arbitrary cut, not at task completion.
        artificial_end_mask=np.ones(shape=len(episode_lengths), dtype=np.bool_),
        # The expert parts are sized for exactly this prediction horizon.
        # A diffusion policy training with any other horizon would mislay its windows.
        intended_prediction_horizon=expert_part_length + 1,
    )
