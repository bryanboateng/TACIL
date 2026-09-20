import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
import wandb
from rich.pretty import pretty_repr
from wandb import Run

from scripts.data.expert.ingest import peg_insertion, pusht
from scripts.environments.registry import EnvironmentName
from scripts.shared.dataset_artifact import DATASET_PROJECT, build_dataset_name
from scripts.shared.episode_store import RewardedEpisodeStore, write_episode_store
from scripts.shared.log import create_logger
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH

logger = create_logger()


@dataclass
class Config:
    environment: EnvironmentName
    source_path: Path


def main() -> None:
    config = tyro.cli(f=Config)
    logger.info(pretty_repr(config))
    _run(config=config)


def _run(*, config: Config) -> None:
    module = _select_module(environment=config.environment)

    logger.info("Reading %s", module.describe_source(source_path=config.source_path))

    episode_store = module.build_episode_store(source_path=config.source_path)
    _log_episode_store(episode_store=episode_store)

    episode_count = len(episode_store.episode_end_indices)

    with wandb.init(
        project=DATASET_PROJECT,
        job_type="expert-data-ingestion",
        dir=OUTPUT_DIRECTORY_PATH,
    ) as run:
        _upload_episode_store(
            episode_store=episode_store,
            dataset_name=build_dataset_name(
                environment=config.environment,
                episode_count=episode_count,
                source_episode_count=episode_count,
            ),
            run=run,
        )

        run.summary["1_episode_count"] = episode_count
        run.summary["2_transition_count"] = len(episode_store.states)
        run.summary["3_state_dimensionality"] = episode_store.states.shape[1]
        run.summary["4_action_dimensionality"] = episode_store.actions.shape[1]


# ---- Private --------------------------------------------


def _select_module(*, environment: EnvironmentName):
    match environment:
        case EnvironmentName.PUSHT:
            return pusht
        case EnvironmentName.PEG_INSERTION:
            return peg_insertion


def _log_episode_store(*, episode_store: RewardedEpisodeStore) -> None:
    episode_lengths = episode_store.episode_lengths  # (episode_count,)
    rewards = episode_store.rewards[
        episode_store.indices_with_successor
    ]  # (transition_count,)

    logger.info(
        "Built %d episodes, %d transitions, state %d, action %d",
        len(episode_store.episode_end_indices),
        len(episode_store.states),
        episode_store.states.shape[1],
        episode_store.actions.shape[1],
    )
    logger.info(
        "Episode length: min %d, max %d, mean %.1f",
        episode_lengths.min(),
        episode_lengths.max(),
        episode_lengths.mean(),
    )
    logger.info(
        "Reward: min %.3f, max %.3f, mean %.3f",
        rewards.min(),
        rewards.max(),
        rewards.mean(),
    )

    if np.isnan(rewards).any():
        raise ValueError(
            "Rewards are missing for transitions that have a successor state."
        )


def _upload_episode_store(
    *, episode_store: RewardedEpisodeStore, dataset_name: str, run: Run
) -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        output_zarr_path = Path(temporary_directory) / "dataset.zarr"

        write_episode_store(
            episode_store=episode_store, save_file_path=output_zarr_path
        )

        artifact = wandb.Artifact(name=dataset_name, type="dataset")
        artifact.add_dir(local_path=str(output_zarr_path))
        run.log_artifact(artifact_or_path=artifact)


if __name__ == "__main__":
    main()
