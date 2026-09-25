import re
import tempfile
from pathlib import Path

import wandb
from wandb import Run

from scripts.shared.episodes import RewardedEpisodes, load_episodes, save_episodes

DATASET_PROJECT = "datasets"


def build_dataset_name(
    *, stem: str, episode_count: int, source_episode_count: int
) -> str:
    return f"{stem}-{episode_count}-of-{source_episode_count}"


def read_dataset_stem(*, dataset_name: str) -> str:
    match = re.compile(r"(?P<stem>.+)-\d+-of-\d+").fullmatch(dataset_name)
    if match is None:
        raise ValueError(
            f"{dataset_name!r} does not end in the "
            f"'-<episode_count>-of-<source_episode_count>' suffix."
        )
    return match.group("stem")


def download_episodes(*, artifact_name: str, run: Run) -> RewardedEpisodes:
    return load_episodes(
        directory_path=(
            Path(
                run.use_artifact(
                    artifact_or_name=f"{DATASET_PROJECT}/{artifact_name}"
                ).download()
            )
        )
    )


def upload_episodes(*, episodes: RewardedEpisodes, dataset_name: str, run: Run) -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        output_directory_path = Path(temporary_directory)

        save_episodes(episodes=episodes, directory_path=output_directory_path)

        artifact = wandb.Artifact(name=dataset_name, type="dataset")
        artifact.add_dir(local_path=str(output_directory_path))
        run.log_artifact(artifact_or_path=artifact)
