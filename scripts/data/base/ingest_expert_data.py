from dataclasses import dataclass
from pathlib import Path

import tyro
import wandb
from rich.pretty import pretty_repr

from scripts.ashared import create_logger
from scripts.shared.data import load_episode_dataset
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH

logger = create_logger()


@dataclass
class Config:
    expert_data_file_path: Path


def main() -> None:
    config = tyro.cli(f=Config, config=[tyro.conf.UsePythonSyntaxForLiteralCollections])
    logger.info(pretty_repr(config))
    _run(expert_data_file_path=config.expert_data_file_path)


def _run(*, expert_data_file_path: Path) -> None:
    with wandb.init(
        project="abc", job_type="ingest-expert-data", dir=OUTPUT_DIRECTORY_PATH
    ) as run:
        artifact = wandb.Artifact(name="expert-dataset", type="dataset")
        artifact.add_dir(local_path=str(expert_data_file_path))
        run.log_artifact(artifact_or_path=artifact)

        episode_dataset = load_episode_dataset(episode_file_path=expert_data_file_path)
        run.summary["episode_count"] = len(episode_dataset.episode_end_indices)
        run.summary["transition_count"] = len(episode_dataset.states)
        run.summary["state_dimensionality"] = episode_dataset.states.shape[1]
        run.summary["action_dimensionality"] = episode_dataset.actions.shape[1]


if __name__ == "__main__":
    main()
