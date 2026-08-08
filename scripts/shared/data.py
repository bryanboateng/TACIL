import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import wandb
import zarr
from numpy import typing as npt
from wandb import Run
from zarr import Array, Group


@dataclass
class EpisodeStore:
    states: npt.NDArray[np.float32]  # (total_timestep_count, state_dimensionality)
    actions: npt.NDArray[np.float32]  # (total_timestep_count, action_dimensionality)
    episode_end_indices: npt.NDArray[np.int64]  # (episode_count,)
    # True for episodes whose end is an artificial mid-task cut (e.g. augmented
    # mini-episodes) rather than a task completion. None means all False.
    artificial_end_mask: npt.NDArray[np.bool_] | None = None  # (episode_count,)
    # The policy prediction horizon the augmented episodes were laid out for
    # (their expert parts are sized against it); policy training must use
    # exactly this horizon or its windows are mislaid. None: no such layout
    # assumption (unaugmented data).
    intended_prediction_horizon: int | None = None


def download_episode_store(*, artifact_name: str, run: Run) -> EpisodeStore:
    return read_episode_store(
        episode_file_path=Path(
            run.use_artifact(artifact_or_name=artifact_name).download()
        )
    )


def upload_episode_store(*, episode_store: EpisodeStore, name: str, run: Run):
    with tempfile.TemporaryDirectory() as temporary_directory:
        output_zarr_path = Path(temporary_directory) / "dataset.zarr"

        _write_episode_store(
            episode_store=episode_store, save_file_path=output_zarr_path
        )

        run_name = run.name
        new_artifact = wandb.Artifact(
            name=f"{name}-{run_name if run_name is not None else datetime.now().isoformat()}",
            type="dataset",
        )
        new_artifact.add_dir(local_path=str(output_zarr_path))
        run.log_artifact(artifact_or_path=new_artifact)


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

    try:
        artificial_end_mask = _get(key="meta/artificial_end_mask", dtype=np.bool_)
    except KeyError:
        artificial_end_mask = None  # datasets created before this field existed

    meta_group = store["meta"]
    intended_prediction_horizon = (
        int(meta_group.attrs["intended_prediction_horizon"])
        if isinstance(meta_group, Group)
        and "intended_prediction_horizon" in meta_group.attrs
        else None
    )

    return EpisodeStore(
        states=_get(key="data/state", dtype=np.float32),
        actions=_get(key="data/action", dtype=np.float32),
        episode_end_indices=_get(key="meta/episode_ends", dtype=np.int64),
        artificial_end_mask=artificial_end_mask,
        intended_prediction_horizon=intended_prediction_horizon,
    )


def combine_episode_stores(
    *, expert_episode_store: EpisodeStore, augmented_episode_store: EpisodeStore
) -> EpisodeStore:
    """Append augmented episodes to the expert data they were derived from.

    The expert store is plain recorded data — no artificial ends, no intended
    prediction horizon. The augmented store carries an artificial end mask,
    and its intended prediction horizon becomes the combined store's. That
    horizon is None when the augmented episodes were laid out for a policy
    that trains on single (state, action) pairs and therefore promise no
    window layout at all.
    """
    if (
        expert_episode_store.artificial_end_mask is not None
        and expert_episode_store.artificial_end_mask.any()
    ):
        raise ValueError(
            "The expert episode store must not contain artificially ended episodes."
        )

    if expert_episode_store.intended_prediction_horizon is not None:
        raise ValueError(
            "The expert episode store must not carry an intended prediction "
            "horizon; it looks like already-augmented data."
        )

    if augmented_episode_store.artificial_end_mask is None:
        raise ValueError(
            "The augmented episode store must carry an artificial end mask."
        )

    expert_timestep_count = expert_episode_store.states.shape[0]
    expert_episode_count = len(expert_episode_store.episode_end_indices)

    return EpisodeStore(
        states=np.concatenate(
            [expert_episode_store.states, augmented_episode_store.states], axis=0
        ),  # (total_timestep_count, state_dimensionality)
        actions=np.concatenate(
            [expert_episode_store.actions, augmented_episode_store.actions], axis=0
        ),  # (total_timestep_count, action_dimensionality)
        episode_end_indices=np.concatenate(
            [
                expert_episode_store.episode_end_indices,
                expert_timestep_count + augmented_episode_store.episode_end_indices,
            ],
            axis=0,
        ),  # (episode_count,)
        artificial_end_mask=np.concatenate(
            [
                np.zeros(shape=expert_episode_count, dtype=np.bool_),
                augmented_episode_store.artificial_end_mask,
            ],
            axis=0,
        ),  # (episode_count,)
        intended_prediction_horizon=augmented_episode_store.intended_prediction_horizon,
    )


# ---- Private --------------------------------------------


def _write_episode_store(*, episode_store: EpisodeStore, save_file_path: Path):
    root = zarr.open_group(str(save_file_path), mode="w")

    data_group = root.create_group("data")
    data_group.create_array("state", data=episode_store.states)
    data_group.create_array("action", data=episode_store.actions)

    meta_group = root.create_group("meta")
    meta_group.create_array("episode_ends", data=episode_store.episode_end_indices)
    meta_group.create_array(
        "artificial_end_mask",
        data=(
            episode_store.artificial_end_mask
            if episode_store.artificial_end_mask is not None
            else np.zeros(len(episode_store.episode_end_indices), dtype=np.bool_)
        ),
    )
    if episode_store.intended_prediction_horizon is not None:
        meta_group.attrs["intended_prediction_horizon"] = int(
            episode_store.intended_prediction_horizon
        )
