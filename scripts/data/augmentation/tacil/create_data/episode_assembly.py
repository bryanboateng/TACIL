from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch

from scripts.data.augmentation.tacil.create_data.backward_rollout import (
    SyntheticTrajectories,
)
from scripts.shared.episode_store import EpisodeStore


@dataclass(frozen=True)
class AugmentationResult:
    @dataclass(frozen=True)
    class OutcomeCounts:
        empty: int  # anchors whose first backward step already broke the budget
        accepted: int

    episode_store: EpisodeStore
    trajectory_lengths: npt.NDArray[np.int64]  # (accepted_count,)
    counts: OutcomeCounts


@torch.no_grad()
def create_augmented_episodes(
    *,
    synthetic_trajectories: SyntheticTrajectories,
    expert_episodes: EpisodeStore,
    anchor_indices: npt.NDArray[np.int64],  # (anchor_count,)
    expert_part_length: int,
) -> AugmentationResult:
    trajectory_lengths: npt.NDArray[np.int64] = (
        synthetic_trajectories.trajectory_lengths.cpu().numpy()
    )  # (anchor_count,)
    accepted_anchor_count = int((trajectory_lengths > 0).sum())
    if accepted_anchor_count == 0:
        raise RuntimeError("Can't create augmented episodes. No anchor was accepted.")

    augmented_episodes = _assemble_episodes(
        expert_episodes=expert_episodes,
        synthetic_trajectories=synthetic_trajectories,
        anchor_index_batch=np.asarray(anchor_indices, dtype=np.int64),
        expert_part_length=expert_part_length,
    )

    return AugmentationResult(
        episode_store=augmented_episodes,
        trajectory_lengths=trajectory_lengths[trajectory_lengths > 0],
        counts=AugmentationResult.OutcomeCounts(
            empty=int(trajectory_lengths.shape[0]) - accepted_anchor_count,
            accepted=accepted_anchor_count,
        ),
    )


# ---- Private --------------------------------------------


def _assemble_episodes(
    *,
    expert_episodes: EpisodeStore,
    synthetic_trajectories: SyntheticTrajectories,
    anchor_index_batch: npt.NDArray[np.int64],  # (anchor_count,)
    expert_part_length: int,
) -> EpisodeStore:
    synthetic_states = synthetic_trajectories.states.cpu().numpy()
    synthetic_actions = synthetic_trajectories.actions.cpu().numpy()

    a = (
        torch.cumsum(input=synthetic_trajectories.trajectory_lengths, dim=0)
        - synthetic_trajectories.trajectory_lengths
    )
    synthetic_part_start_index_batch = a.cpu().numpy()  # (anchor_count,)

    trajectory_length_batch = (
        synthetic_trajectories.trajectory_lengths.cpu().numpy()
    )  # (anchor_count,)

    accepted_anchor_mask = trajectory_length_batch > 0  # (anchor_count,)

    state_trajectories_per_episode: list[np.ndarray] = []
    action_trajectories_per_episode: list[np.ndarray] = []
    episode_lengths: list[int] = []

    for trajectory_length, synthetic_part_start_index, anchor_index in zip(
        trajectory_length_batch[accepted_anchor_mask],
        synthetic_part_start_index_batch[accepted_anchor_mask],
        anchor_index_batch[accepted_anchor_mask],
    ):
        synthetic_part_slice = slice(
            synthetic_part_start_index, synthetic_part_start_index + trajectory_length
        )

        synthetic_state_trajectory = synthetic_states[
            synthetic_part_slice
        ]  # (trajectory_length, state_dimensionality)
        synthetic_action_trajectory = synthetic_actions[
            synthetic_part_slice
        ]  # (trajectory_length, action_dimensionality)

        expert_part_slice = slice(anchor_index, anchor_index + expert_part_length)

        state_trajectories_per_episode.append(
            np.concatenate(
                [synthetic_state_trajectory, expert_episodes.states[expert_part_slice]],
                axis=0,
            )
        )
        action_trajectories_per_episode.append(
            np.concatenate(
                [
                    synthetic_action_trajectory,
                    expert_episodes.actions[expert_part_slice],
                ],
                axis=0,
            )
        )
        episode_lengths.append(int(trajectory_length) + expert_part_length)

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
    )
