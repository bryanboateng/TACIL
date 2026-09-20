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
class EpisodeStore:
    states: npt.NDArray[np.float32]  # (total_timestep_count, state_dimensionality)
    actions: npt.NDArray[np.float32]  # (total_timestep_count, action_dimensionality)
    episode_end_indices: npt.NDArray[np.int64]  # (episode_count,)

    @property
    def episode_start_indices(self) -> npt.NDArray[np.int64]:  # (episode_count,)
        return np.concatenate(
            (
                np.zeros(shape=1, dtype=self.episode_end_indices.dtype),
                self.episode_end_indices[:-1],
            )
        )

    @property
    def episode_lengths(self) -> npt.NDArray[np.int64]:  # (episode_count,)
        return np.diff(a=self.episode_end_indices, prepend=0)

    @property
    def indices_with_successor(self) -> npt.NDArray[np.int64]:  # (transition_count,)
        has_successor = np.ones(
            shape=len(self.states), dtype=np.bool_
        )  # (total_timestep_count,)
        has_successor[self.episode_end_indices - 1] = False
        return np.flatnonzero(has_successor)


@dataclass
class RewardedEpisodeStore(EpisodeStore):
    rewards: npt.NDArray[np.float32]  # (total_timestep_count,)


def read_episode_store(*, episode_file_path: Path) -> EpisodeStore:
    reader = _Reader(episode_file_path=episode_file_path)
    return EpisodeStore(
        states=reader.read(key=_STATE_KEY, dtype=np.float32),
        actions=reader.read(key=_ACTION_KEY, dtype=np.float32),
        episode_end_indices=reader.read(key=_EPISODE_END_KEY, dtype=np.int64),
    )


def read_rewarded_episode_store(*, episode_file_path: Path) -> RewardedEpisodeStore:
    reader = _Reader(episode_file_path=episode_file_path)

    if _REWARD_KEY not in reader.store:
        raise KeyError("Dataset has no rewards. Only datasets.")

    return RewardedEpisodeStore(
        states=reader.read(key=_STATE_KEY, dtype=np.float32),
        actions=reader.read(key=_ACTION_KEY, dtype=np.float32),
        episode_end_indices=reader.read(key=_EPISODE_END_KEY, dtype=np.int64),
        rewards=reader.read(key=_REWARD_KEY, dtype=np.float32),
    )


def write_episode_store(*, episode_store: EpisodeStore, save_file_path: Path) -> None:
    root = zarr.open_group(str(save_file_path), mode="w")

    data_group = root.create_group("data")
    data_group.create_array("state", data=episode_store.states)
    data_group.create_array("action", data=episode_store.actions)
    if isinstance(episode_store, RewardedEpisodeStore):
        data_group.create_array("reward", data=episode_store.rewards)

    meta_group = root.create_group("meta")
    meta_group.create_array("episode_ends", data=episode_store.episode_end_indices)


# ---- Private --------------------------------------------


class _Reader:
    def __init__(self, *, episode_file_path: Path) -> None:
        store = zarr.open(store=episode_file_path, mode="r")

        if not isinstance(store, Group):
            raise TypeError(
                f"Expected zarr.Group at {episode_file_path}, "
                f"got {type(store).__name__}"
            )

        self.store = store

    def read(self, *, key: str, dtype: object) -> np.ndarray:
        try:
            group_member = self.store[key]
        except KeyError as exc:
            raise KeyError(f"Missing dataset entry {key!r}") from exc

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
