import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tyro
import wandb
import zarr
from rich.pretty import pretty_repr
from wandb import Run

from scripts.shared.dataset_artifact import (
    DATASET_PROJECT,
    build_dataset_name,
    download_episode_store,
)
from scripts.shared.episode_store import EpisodeStore
from scripts.shared.log import create_logger
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH

logger = create_logger()


@dataclass
class Config:
    source_dataset_artifact: str
    episode_count: int

    def __post_init__(self):
        if self.episode_count < 1:
            raise ValueError(
                f"episode_count must be positive, but it is {self.episode_count}."
            )


def main():
    config = tyro.cli(f=Config)
    logger.info(pretty_repr(config))
    _run(config=config)


def _run(*, config: Config):
    with wandb.init(
        project=DATASET_PROJECT,
        job_type="dataset-subsampling",
        config=asdict(config),
        dir=OUTPUT_DIRECTORY_PATH,
    ) as run:
        expert_episode_store = download_episode_store(
            artifact_name=config.source_dataset_artifact, run=run
        )

        source_episode_count = len(expert_episode_store.episode_end_indices)
        source_name = config.source_dataset_artifact.split(":", maxsplit=1)[0]

        if not source_name.endswith(f"-of-{source_episode_count}"):
            raise ValueError(
                f"Subsampling needs a full dataset, but {source_name!r} holds "
                f"{source_episode_count} episodes."
            )

        subsampled_episode_store = _subsample_episodes(
            episode_store=expert_episode_store, episode_count=config.episode_count
        )

        _log_dataset_statistics(
            expert_episode_store=expert_episode_store,
            subsampled_episode_store=subsampled_episode_store,
            run=run,
        )

        _upload_episode_store(
            subsampled_episode_store=subsampled_episode_store,
            dataset_name=build_dataset_name(
                environment=source_name.removeprefix("dataset-").split("-")[0],
                episode_count=len(subsampled_episode_store.episode_end_indices),
                source_episode_count=source_episode_count,
            ),
            run=run,
        )


def _subsample_episodes(
    *, episode_store: EpisodeStore, episode_count: int
) -> EpisodeStore:

    source_episode_count = len(episode_store.episode_end_indices)

    selected_episode_indices = np.sort(
        a=np.random.default_rng(seed=42).choice(
            a=source_episode_count, size=episode_count, replace=False
        )
    )  # (selected_episode_count,)

    episode_start_indices = np.concatenate(
        (
            np.zeros(shape=1, dtype=episode_store.episode_end_indices.dtype),
            episode_store.episode_end_indices[:-1],
        )
    )  # (episode_count,)

    selected_transition_indices_list = []
    subsampled_episode_end_indices = []
    cumulative_transition_count = 0

    for selected_episode_index in selected_episode_indices:
        if not isinstance(selected_episode_index, int):
            raise TypeError(
                f"Expected int, got {type(selected_episode_index).__name__}"
            )

        start_index: int = episode_start_indices[selected_episode_index]
        end_index: int = episode_store.episode_end_indices[selected_episode_index]

        episode_transition_indices = np.arange(
            start_index, stop=end_index
        )  # (episode_length,)

        selected_transition_indices_list.append(episode_transition_indices)

        cumulative_transition_count += len(episode_transition_indices)
        subsampled_episode_end_indices.append(cumulative_transition_count)

    selected_transition_indices_np_array = np.concatenate(
        selected_transition_indices_list
    )  # (selected_transition_count,)

    return EpisodeStore(
        states=episode_store.states[
            selected_transition_indices_np_array
        ],  # (selected_transition_count, state_dimensionality)
        actions=episode_store.actions[
            selected_transition_indices_np_array
        ],  # (selected_transition_count, action_dimensionality)
        episode_end_indices=np.asarray(
            subsampled_episode_end_indices
        ),  # (selected_episode_count,)
    )


def _log_dataset_statistics(
    expert_episode_store: EpisodeStore, subsampled_episode_store: EpisodeStore, run: Run
):
    expert_episode_count = len(expert_episode_store.episode_end_indices)
    expert_transition_count = len(expert_episode_store.states)

    run.summary["1_source_dataset/1_episode_count"] = expert_episode_count
    run.summary["1_source_dataset/2_transition_count"] = expert_transition_count

    selected_episode_count = len(subsampled_episode_store.episode_end_indices)
    selected_transition_count = len(subsampled_episode_store.states)

    run.summary["2_subsampled_dataset/1_episode_count"] = selected_episode_count
    run.summary["2_subsampled_dataset/3_episode_fraction_of_source"] = (
        selected_episode_count / expert_episode_count
    )
    run.summary["2_subsampled_dataset/2_transition_count"] = selected_transition_count
    run.summary["2_subsampled_dataset/4_transition_fraction_of_source"] = (
        selected_transition_count / expert_transition_count
    )


def _upload_episode_store(
    *, subsampled_episode_store: EpisodeStore, dataset_name: str, run: Run
):
    with tempfile.TemporaryDirectory() as temporary_directory:
        output_zarr_path = Path(temporary_directory) / "dataset.zarr"

        _write_episode_store(
            episode_store=subsampled_episode_store, save_file_path=output_zarr_path
        )

        new_artifact = wandb.Artifact(name=dataset_name, type="dataset")
        new_artifact.add_dir(local_path=str(output_zarr_path))
        run.log_artifact(artifact_or_path=new_artifact)


def _write_episode_store(*, episode_store: EpisodeStore, save_file_path: Path):
    root = zarr.open_group(str(save_file_path), mode="w")

    data_group = root.create_group("data")
    data_group.create_array("state", data=episode_store.states)
    data_group.create_array("action", data=episode_store.actions)

    meta_group = root.create_group("meta")
    meta_group.create_array("episode_ends", data=episode_store.episode_end_indices)


if __name__ == "__main__":
    main()
