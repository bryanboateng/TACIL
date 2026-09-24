from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr
from numpy import typing as npt
from zarr import Array, Group

_STATE_KEY = "data/state"
_ACTION_KEY = "data/action"
_REWARD_KEY = "data/reward"
_EPISODE_END_KEY = "meta/episode_ends"


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


def read_episodes(*, episode_file_path: Path) -> RewardedEpisodes:
    reader = _Reader(episode_file_path=episode_file_path)
    return RewardedEpisodes(
        states=reader.read(key=_STATE_KEY, dtype=np.float32),
        actions=reader.read(key=_ACTION_KEY, dtype=np.float32),
        episode_end_indices=reader.read(key=_EPISODE_END_KEY, dtype=np.int64),
        rewards=reader.read(key=_REWARD_KEY, dtype=np.float32),
    )


def write_episodes(*, episodes: RewardedEpisodes, save_file_path: Path) -> None:
    root = zarr.open_group(str(save_file_path), mode="w")

    data_group = root.create_group("data")
    data_group.create_array("state", data=episodes.states)
    data_group.create_array("action", data=episodes.actions)
    data_group.create_array("reward", data=episodes.rewards)

    meta_group = root.create_group("meta")
    meta_group.create_array("episode_ends", data=episodes.episode_end_indices)


# ---- Private --------------------------------------------


class _Reader:
    def __init__(self, *, episode_file_path: Path) -> None:
        store = zarr.open(store=episode_file_path, mode="r")

        if not isinstance(store, Group):
            raise TypeError(
                f"Expected zarr.Group at {episode_file_path}, "
                f"got {type(store).__name__}"
            )

        self.episode_file_path = episode_file_path
        self.store = store

    def read(self, *, key: str, dtype: object) -> np.ndarray:
        try:
            group_member = self.store[key]
        except KeyError as exc:
            raise KeyError(
                f"Missing dataset entry {key!r} in {self.episode_file_path}"
            ) from exc

        if not isinstance(group_member, Array):
            raise TypeError(f"{key!r} is not a zarr.Array")

        data = group_member[:]

        if not isinstance(data, np.ndarray):
            raise TypeError(
                f"{key!r} did not load as np.ndarray (got {type(data).__name__})"
            )

        expected_dtype = np.dtype(dtype)

        if data.dtype != expected_dtype:
            raise TypeError(
                f"{key!r} has dtype {data.dtype}, expected {expected_dtype}"
            )

        return data
