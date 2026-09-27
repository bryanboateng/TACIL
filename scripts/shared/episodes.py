from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy import typing as npt

EPISODES_ARCHIVE_NAME = "episodes.npz"


@dataclass
class Episodes:
    states: npt.NDArray[np.float32]  # (total_timestep_count, state_dimensionality)
    actions: npt.NDArray[np.float32]  # (total_timestep_count, action_dimensionality)
    episode_end_indices: npt.NDArray[np.int64]  # (episode_count,)


@dataclass
class RewardedEpisodes:
    states: npt.NDArray[np.float32]  # (total_timestep_count, state_dimensionality)
    actions: npt.NDArray[np.float32]  # (total_timestep_count, action_dimensionality)
    episode_end_indices: npt.NDArray[np.int64]  # (episode_count,)
    rewards: npt.NDArray[np.float32]  # (total_timestep_count,)


def calculate_episode_start_indices(
    *,
    episode_end_indices: npt.NDArray[np.int64],  # (episode_count,)
) -> npt.NDArray[np.int64]:  # (episode_count,)
    return np.concatenate(
        (np.zeros(shape=1, dtype=episode_end_indices.dtype), episode_end_indices[:-1])
    )


def calculate_episode_lengths(
    *,
    episode_end_indices: npt.NDArray[np.int64],  # (episode_count,)
) -> npt.NDArray[np.int64]:  # (episode_count,)
    return np.diff(a=episode_end_indices, prepend=0)


def calculate_indices_with_successor(
    *,
    episode_end_indices: npt.NDArray[np.int64],  # (episode_count,)
    total_timestep_count: int,
) -> npt.NDArray[np.int64]:  # (transition_count,)
    has_successor = np.ones(
        shape=total_timestep_count, dtype=np.bool_
    )  # (total_timestep_count,)
    has_successor[episode_end_indices - 1] = False
    return np.flatnonzero(has_successor)


def drop_rewards(*, episodes: RewardedEpisodes) -> Episodes:
    return Episodes(
        states=episodes.states,
        actions=episodes.actions,
        episode_end_indices=episodes.episode_end_indices,
    )


def load_episodes(*, directory_path: Path) -> RewardedEpisodes:
    with np.load(file=directory_path / EPISODES_ARCHIVE_NAME) as episodes_archive:
        return RewardedEpisodes(
            states=episodes_archive["states"],
            actions=episodes_archive["actions"],
            episode_end_indices=episodes_archive["episode_end_indices"],
            rewards=episodes_archive["rewards"],
        )


def save_episodes(*, episodes: RewardedEpisodes, directory_path: Path) -> None:
    np.savez_compressed(
        file=directory_path / EPISODES_ARCHIVE_NAME,
        states=episodes.states,
        actions=episodes.actions,
        episode_end_indices=episodes.episode_end_indices,
        rewards=episodes.rewards,
    )
