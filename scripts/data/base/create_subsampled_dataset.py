import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tyro
import wandb
import zarr
from rich.pretty import pretty_repr
from wandb import Run

from scripts.data.base.episode_store import EpisodeStore, download_episode_store
from scripts.shared.log import create_logger
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH

logger = create_logger()


@dataclass
class Config:
    wandb_project: str

    fraction: float
    seed: int

    def __post_init__(self):
        if not 0.0 <= self.fraction <= 1.0:
            raise ValueError(f"fraction must be in [0, 1], got {self.fraction}")


def main():
    config = tyro.cli(f=Config, config=[tyro.conf.UsePythonSyntaxForLiteralCollections])
    logger.info(pretty_repr(config))
    _run(config=config)


def _run(*, config: Config):
    rng = np.random.default_rng(seed=config.seed)

    with wandb.init(
        project=config.wandb_project,
        job_type="create-subsampled-dataset",
        config=asdict(config),
        dir=OUTPUT_DIRECTORY_PATH,
    ) as run:
        expert_episode_store = download_episode_store(
            artifact_name="expert-dataset:latest", run=run
        )

        subsampled_episode_store = _subsample_episodes(
            episode_store=expert_episode_store, fraction=config.fraction, rng=rng
        )

        _log_dataset_statistics(
            expert_episode_store=expert_episode_store,
            subsampled_episode_store=subsampled_episode_store,
            run=run,
        )

        _upload_episode_store(
            subsampled_episode_store=subsampled_episode_store,
            fraction=config.fraction,
            run=run,
        )


def _subsample_episodes(
    *, episode_store: EpisodeStore, fraction: float, rng: np.random.Generator
) -> EpisodeStore:

    episode_count = len(episode_store.episode_end_indices)

    selected_episode_indices = np.sort(
        a=rng.choice(a=episode_count, size=int(episode_count * fraction), replace=False)
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
        selected_episode_index: int
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

    run.summary["source_dataset"] = {
        "episode_count": expert_episode_count,
        "transitions_count": expert_transition_count,
    }

    selected_episode_count = len(subsampled_episode_store.episode_end_indices)
    selected_transition_count = len(subsampled_episode_store.states)

    run.summary["subsampled_dataset"] = {
        "episodes": {
            "count": selected_episode_count,
            "fraction_of_source": selected_episode_count / expert_episode_count,
        },
        "transitions": {
            "count": selected_transition_count,
            "fraction_of_source": (selected_transition_count / expert_transition_count),
        },
    }


def _upload_episode_store(
    *, subsampled_episode_store: EpisodeStore, fraction: float, run: Run
):
    with tempfile.TemporaryDirectory() as temporary_directory:
        output_zarr_path = Path(temporary_directory) / "dataset.zarr"

        _write_episode_store(
            episode_store=subsampled_episode_store, save_file_path=output_zarr_path
        )

        new_artifact = wandb.Artifact(
            name=f"subsampled-dataset-{fraction}", type="dataset"
        )
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
