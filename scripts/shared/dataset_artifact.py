from pathlib import Path

from wandb import Run

from scripts.environments.registry import EnvironmentName
from scripts.shared.episode_store import EpisodeStore, read_episode_store

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


def download_episode_store(*, artifact_name: str, run: Run) -> EpisodeStore:
    return read_episode_store(
        episode_file_path=Path(
            run.use_artifact(
                artifact_or_name=f"{DATASET_PROJECT}/{artifact_name}"
            ).download()
        )
    )
