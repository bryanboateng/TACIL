import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr
from numpy import typing as npt
from wandb import Artifact
from zarr import Array, Group


@dataclass
class EpisodeDataset:
    states: npt.NDArray[np.float32]
    actions: npt.NDArray[np.float32]
    episode_end_indices: npt.NDArray[np.int64]


def save_episode_dataset(*, episode_dataset: EpisodeDataset, save_file_path: Path):
    root = zarr.open(store=save_file_path, mode="w")

    assert isinstance(root, Group)

    root.create_array(
        "data/state",
        data=episode_dataset.states,
        chunks=(
            min(1024, len(episode_dataset.states)),
            episode_dataset.states.shape[1],
        ),
    )

    root.create_array(
        "data/action",
        data=episode_dataset.actions,
        chunks=(
            min(1024, len(episode_dataset.actions)),
            episode_dataset.actions.shape[1],
        ),
    )

    root.create_array(
        "meta/episode_ends",
        data=episode_dataset.episode_end_indices,
        chunks=(min(1024, len(episode_dataset.episode_end_indices)),),
    )


def load_episode_dataset_from_artifact(
    *, artifact: Artifact, logger: logging.Logger
) -> EpisodeDataset:
    artifact_content_path = Path(artifact.download())

    logger.info("Loading dataset from path %s ...", artifact_content_path)

    episode_dataset = _load_episode_dataset(episode_file_path=artifact_content_path)

    logger.info("Loading dataset completed.")
    logger.info("# Transitions: %d", len(episode_dataset.states))
    logger.info("# Episodes: %d", len(episode_dataset.episode_end_indices))

    return episode_dataset


def _load_episode_dataset(*, episode_file_path: Path) -> EpisodeDataset:
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
