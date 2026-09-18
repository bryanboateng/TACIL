from dataclasses import dataclass
from pathlib import Path

import tyro
import wandb
from rich.pretty import pretty_repr

from scripts.environments.registry import EnvironmentName
from scripts.shared.dataset_artifact import DATASET_PROJECT, build_dataset_name
from scripts.shared.episode_store import read_episode_store
from scripts.shared.log import create_logger
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH

logger = create_logger()


@dataclass
class Config:
    environment: EnvironmentName
    expert_data_file_path: Path


def main() -> None:
    config = tyro.cli(f=Config)
    logger.info(pretty_repr(config))
    _run(config=config)


def _run(*, config: Config) -> None:
    with wandb.init(
        project=DATASET_PROJECT,
        job_type="expert-data-ingestion",
        dir=OUTPUT_DIRECTORY_PATH,
    ) as run:
        episode_store = read_episode_store(
            episode_file_path=config.expert_data_file_path
        )

        episode_count = len(episode_store.episode_end_indices)
        artifact = wandb.Artifact(
            name=build_dataset_name(
                environment=config.environment,
                episode_count=episode_count,
                source_episode_count=episode_count,
            ),
            type="dataset",
        )
        artifact.add_dir(local_path=str(config.expert_data_file_path))
        run.log_artifact(artifact_or_path=artifact)

        run.summary["1_episode_count"] = episode_count
        run.summary["2_transition_count"] = len(episode_store.states)
        run.summary["3_state_dimensionality"] = episode_store.states.shape[1]
        run.summary["4_action_dimensionality"] = episode_store.actions.shape[1]
        run.summary["5_carries_rewards"] = episode_store.rewards is not None


if __name__ == "__main__":
    main()
