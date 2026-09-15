from pathlib import Path

from wandb import Run

from scripts.shared.episode_store import EpisodeStore, read_episode_store

DATASET_PROJECT = "datasets"


def build_dataset_name(
    *, environment: str, episode_count: int, source_episode_count: int
) -> str:
    return f"dataset-{environment}-{episode_count}-of-{source_episode_count}"


def download_episode_store(*, artifact_name: str, run: Run) -> EpisodeStore:
    return read_episode_store(
        episode_file_path=Path(
            run.use_artifact(
                artifact_or_name=f"{DATASET_PROJECT}/{artifact_name}"
            ).download()
        )
    )
