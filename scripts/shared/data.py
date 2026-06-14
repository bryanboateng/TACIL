import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wandb
import zarr
from numpy import typing as npt
from wandb import Artifact, Run
from zarr import Array, Group


@dataclass
class EpisodeDataset:
    states: npt.NDArray[np.float32]  # (total_timestep_count, state_dimensionality)
    actions: npt.NDArray[np.float32]  # (total_timestep_count, action_dimensionality)
    episode_end_indices: npt.NDArray[np.int64]  # (episode_count,)


def save_episode_dataset(*, episode_dataset: EpisodeDataset, save_file_path: Path):
    root = zarr.open_group(str(save_file_path), mode="w")

    data_group = root.create_group("data")
    data_group.create_array("state", data=episode_dataset.states)
    data_group.create_array("action", data=episode_dataset.actions)

    meta_group = root.create_group("meta")
    meta_group.create_array("episode_ends", data=episode_dataset.episode_end_indices)


def load_episode_dataset(*, episode_file_path: Path) -> EpisodeDataset:
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

    return EpisodeDataset(
        states=_get(key="data/state", dtype=np.float32),
        actions=_get(key="data/action", dtype=np.float32),
        episode_end_indices=_get(key="meta/episode_ends", dtype=np.int64),
    )


def upload_episode_dataset(episode_dataset: EpisodeDataset, name: str, run: Run):
    with tempfile.TemporaryDirectory() as temporary_directory:
        output_zarr_path = Path(temporary_directory) / "dataset.zarr"

        save_episode_dataset(
            episode_dataset=episode_dataset, save_file_path=output_zarr_path
        )

        new_artifact = wandb.Artifact(name=f"{name}-{run.name}", type="dataset")
        new_artifact.add_dir(local_path=str(output_zarr_path))
        run.log_artifact(artifact_or_path=new_artifact)
