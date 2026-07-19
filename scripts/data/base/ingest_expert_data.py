from dataclasses import dataclass
from pathlib import Path

import tyro
import wandb
from rich.pretty import pretty_repr

from scripts.shared.data import load_episode_store
from scripts.shared.log import create_logger
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH
from scripts.shared.wandb_config import WandbConfig

logger = create_logger()


@dataclass
class Config:
    wandb: WandbConfig
    expert_data_file_path: Path


def main() -> None:
    config = tyro.cli(f=Config, config=[tyro.conf.UsePythonSyntaxForLiteralCollections])
    logger.info(pretty_repr(config))
    _run(config=config)


def _run(*, config: Config) -> None:
    with wandb.init(
        project=config.wandb.project,
        notes=config.wandb.notes,
        job_type="ingest-expert-data",
        dir=OUTPUT_DIRECTORY_PATH,
    ) as run:
        artifact = wandb.Artifact(name="expert-dataset", type="dataset")
        artifact.add_dir(local_path=str(config.expert_data_file_path))
        run.log_artifact(artifact_or_path=artifact)

        episode_store = load_episode_store(
            episode_file_path=config.expert_data_file_path
        )
        run.summary["episode_count"] = len(episode_store.episode_end_indices)
        run.summary["transition_count"] = len(episode_store.states)
        run.summary["state_dimensionality"] = episode_store.states.shape[1]
        run.summary["action_dimensionality"] = episode_store.actions.shape[1]


if __name__ == "__main__":
    main()
