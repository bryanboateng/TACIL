import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import tyro
import wandb
import zarr
from rich.pretty import pretty_repr
from zarr import Array, Group

from scripts.paths import OUTPUT_ROOT
from scripts.shared import compute_config_hash, create_logger

logger = create_logger()


@dataclass
class Config:
    fraction: float
    seed: int = 42


@dataclass
class DemonstrationData:
    states: npt.NDArray[np.float32]
    actions: npt.NDArray[np.float32]
    episode_end_indices: npt.NDArray[np.int64]


def main():
    config = tyro.cli(f=Config, config=[tyro.conf.UsePythonSyntaxForLiteralCollections])
    logger.info(pretty_repr(config))
    _run(config=config)


def _run(config: Config):
    rng = np.random.default_rng(seed=config.seed)

    with wandb.init(
        project="abc",
        job_type="subsample-dataset",
        config=asdict(config),
        dir=OUTPUT_ROOT,
    ) as run:
        artifact = run.use_artifact(artifact_or_name="expert-dataset:latest")

        artifact_content_path = Path(artifact.download())

        logger.info("Loading dataset from path %s ...", artifact_content_path)

        demonstration_data = _load_zarr(zarr_file_path=artifact_content_path)

        logger.info("Loading dataset completed.")
        logger.info("# Transitions: %d", len(demonstration_data.states))
        logger.info("# Episodes: %d", len(demonstration_data.episode_end_indices))

        logger.info("Subsampling dataset with %g%% ...", config.fraction * 100)
        subsampled_episodes = _subsample_episodes(
            demonstration_data=demonstration_data, fraction=config.fraction, rng=rng
        )

        logger.info("Subsampling dataset completed.")
        logger.info("# Transitions: %d", len(subsampled_episodes.states))
        logger.info("# Episodes: %d", len(subsampled_episodes.episode_end_indices))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_zarr_path = Path(tmpdir) / "dataset.zarr"

            _write_zarr(
                zarr_file_path=output_zarr_path, demonstration_data=subsampled_episodes
            )

            new_artifact = wandb.Artifact(
                name=f"subsampled-dataset-{compute_config_hash(config=asdict(config))}",
                type="dataset",
            )

            new_artifact.add_dir(str(output_zarr_path))

            run.log_artifact(new_artifact)


def _subsample_episodes(
    demonstration_data: DemonstrationData, fraction: float, rng: np.random.Generator
) -> DemonstrationData:
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    episode_end_indices = demonstration_data.episode_end_indices

    episode_count = len(episode_end_indices)

    selected_episodes_count = int(episode_count * fraction)

    if selected_episodes_count < 1:
        raise ValueError(
            f"fraction={fraction} results in selecting 0 episodes "
            f"from episode_count={episode_count}. "
            f"Increase the fraction."
        )

    selected_episode_indices = np.sort(
        a=rng.choice(a=episode_count, size=selected_episodes_count, replace=False)
    )

    episode_start_indices = np.concatenate(
        (np.zeros(shape=1, dtype=episode_end_indices.dtype), episode_end_indices[:-1])
    )

    selected_transition_indices_list = []

    subsampled_episode_end_indices = []

    cumulative_transition_count = 0

    for selected_episode_index in selected_episode_indices:
        selected_episode_index: int
        start_index: int = episode_start_indices[selected_episode_index]
        end_index: int = episode_end_indices[selected_episode_index]

        episode_transition_indices = np.arange(start_index, stop=end_index)

        selected_transition_indices_list.append(episode_transition_indices)

        cumulative_transition_count += len(episode_transition_indices)
        subsampled_episode_end_indices.append(cumulative_transition_count)

    selected_transition_indices_np_array = np.concatenate(
        selected_transition_indices_list
    )

    return DemonstrationData(
        states=demonstration_data.states[selected_transition_indices_np_array],
        actions=demonstration_data.actions[selected_transition_indices_np_array],
        episode_end_indices=np.asarray(subsampled_episode_end_indices),
    )


def _write_zarr(zarr_file_path: Path, demonstration_data: DemonstrationData):
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


def _load_zarr(zarr_file_path: Path) -> DemonstrationData:
    store = zarr.open(store=zarr_file_path, mode="r")

    if not isinstance(store, Group):
        raise TypeError(
            f"Expected zarr.Group at {zarr_file_path}, got {type(store).__name__}"
        )

    def _get(key: str, dtype: object) -> np.ndarray:
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


if __name__ == "__main__":
    main()
