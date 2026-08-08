from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr
from numpy import typing as npt
from wandb import Run
from zarr import Array, Group


@dataclass
class TransitionBatch:
    states: npt.NDArray[np.float32]  # (transition_count, state_dimensionality)
    actions: npt.NDArray[np.float32]  # (transition_count, action_dimensionality)


@dataclass
class EpisodeStore:
    states: npt.NDArray[np.float32]  # (total_timestep_count, state_dimensionality)
    actions: npt.NDArray[np.float32]  # (total_timestep_count, action_dimensionality)
    episode_end_indices: npt.NDArray[np.int64]  # (episode_count,)


def download_episode_store(*, artifact_name: str, run: Run) -> EpisodeStore:
    return read_episode_store(
        episode_file_path=Path(
            run.use_artifact(artifact_or_name=artifact_name).download()
        )
    )


def read_episode_store(*, episode_file_path: Path) -> EpisodeStore:
    store = zarr.open(store=episode_file_path, mode="r")

    if not isinstance(store, Group):
        raise TypeError(
            f"Expected zarr.Group at {episode_file_path}, got {type(store).__name__}"
        )

    def _get(*, key: str, dtype: object) -> np.ndarray:
        try:
            group_member = store[key]
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

    return EpisodeStore(
        states=_get(key="data/state", dtype=np.float32),
        actions=_get(key="data/action", dtype=np.float32),
        episode_end_indices=_get(key="meta/episode_ends", dtype=np.int64),
    )
