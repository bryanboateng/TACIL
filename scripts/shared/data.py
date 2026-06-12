import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr
from numpy import typing as npt
from wandb import Artifact
from zarr import Array, Group


@dataclass
class DemonstrationData:
    states: npt.NDArray[np.float32]
    actions: npt.NDArray[np.float32]
    episode_end_indices: npt.NDArray[np.int64]


def write_zarr(*, zarr_file_path: Path, demonstration_data: DemonstrationData):
    root = zarr.open(store=zarr_file_path, mode="w")

    assert isinstance(root, Group)

    root.create_array(
        "data/state",
        data=demonstration_data.states,
        chunks=(
            min(1024, len(demonstration_data.states)),
            demonstration_data.states.shape[1],
        ),
    )

    root.create_array(
        "data/action",
        data=demonstration_data.actions,
        chunks=(
            min(1024, len(demonstration_data.actions)),
            demonstration_data.actions.shape[1],
        ),
    )

    root.create_array(
        "meta/episode_ends",
        data=demonstration_data.episode_end_indices,
        chunks=(min(1024, len(demonstration_data.episode_end_indices)),),
    )


def load_zarr(*, zarr_file_path: Path) -> DemonstrationData:
    store = zarr.open(store=zarr_file_path, mode="r")

    if not isinstance(store, Group):
        raise TypeError(
            f"Expected zarr.Group at {zarr_file_path}, got {type(store).__name__}"
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

    return DemonstrationData(
        states=_get(key="data/state", dtype=np.float32),
        actions=_get(key="data/action", dtype=np.float32),
        episode_end_indices=_get(key="meta/episode_ends", dtype=np.int64),
    )


def load_demonstration_data_from_artifact(
    *, artifact: Artifact, logger: logging.Logger
) -> DemonstrationData:
    artifact_content_path = Path(artifact.download())

    logger.info("Loading dataset from path %s ...", artifact_content_path)

    demonstration_data = load_zarr(zarr_file_path=artifact_content_path)

    logger.info("Loading dataset completed.")
    logger.info("# Transitions: %d", len(demonstration_data.states))
    logger.info("# Episodes: %d", len(demonstration_data.episode_end_indices))

    return demonstration_data
