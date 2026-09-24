import tempfile
from pathlib import Path

import wandb
from wandb import Run

from scripts.environments.registry import EnvironmentName
from scripts.shared.episodes import Episodes, read_episodes, write_episodes

DATASET_PROJECT = "datasets"


def build_dataset_name_prefix(*, environment: EnvironmentName) -> str:
    return f"dataset-{environment.value}-"


def build_dataset_name(
    *, environment: EnvironmentName, episode_count: int, source_episode_count: int
) -> str:
    return (
        f"{build_dataset_name_prefix(environment=environment)}"
        f"{episode_count}-of-{source_episode_count}"
    )


def read_environment_name(*, dataset_name: str) -> EnvironmentName:
    for environment in EnvironmentName:
        if dataset_name.startswith(build_dataset_name_prefix(environment=environment)):
            return environment
    raise ValueError(
        f"{dataset_name!r} names no known environment. Known prefixes: "
        f"{[build_dataset_name_prefix(environment=name) for name in EnvironmentName]}"
    )


def download_episodes(*, artifact_name: str, run: Run) -> Episodes:
    return read_episodes(
        episode_file_path=_download(artifact_name=artifact_name, run=run)
    )


def upload_episodes(*, episodes: Episodes, dataset_name: str, run: Run) -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        output_zarr_path = Path(temporary_directory) / "dataset.zarr"

        write_episodes(episodes=episodes, save_file_path=output_zarr_path)

        artifact = wandb.Artifact(name=dataset_name, type="dataset")
        artifact.add_dir(local_path=str(output_zarr_path))
        run.log_artifact(artifact_or_path=artifact)


# ---- Private --------------------------------------------


def _download(*, artifact_name: str, run: Run) -> Path:
    return Path(
        run.use_artifact(
            artifact_or_name=f"{DATASET_PROJECT}/{artifact_name}"
        ).download()
    )
