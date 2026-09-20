from dataclasses import dataclass

import numpy as np
import tyro
import wandb
from rich.pretty import pretty_repr
from wandb import Run

from scripts.data.expert.ingest import peg_insertion, pusht
from scripts.environments.registry import EnvironmentName
from scripts.shared.dataset_artifact import (
    DATASET_PROJECT,
    build_dataset_name,
    upload_episode_store,
)
from scripts.shared.distribution_summary import summarize_distribution
from scripts.shared.episode_store import RewardedEpisodeStore
from scripts.shared.log import create_logger
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH

logger = create_logger()


@dataclass
class Config:
    environment: EnvironmentName


def main() -> None:
    config = tyro.cli(f=Config)
    logger.info(pretty_repr(config))
    _run(config=config)


def _run(*, config: Config) -> None:
    match config.environment:
        case EnvironmentName.PUSH_T:
            module = pusht
        case EnvironmentName.PEG_INSERTION:
            module = peg_insertion

    episode_store = module.build_episode_store()
    _check_rewards(episode_store=episode_store)

    episode_count = len(episode_store.episode_end_indices)

    with wandb.init(
        project=DATASET_PROJECT,
        job_type="expert-data-ingestion",
        dir=OUTPUT_DIRECTORY_PATH,
    ) as run:
        upload_episode_store(
            episode_store=episode_store,
            dataset_name=build_dataset_name(
                environment=config.environment,
                episode_count=episode_count,
                source_episode_count=episode_count,
            ),
            run=run,
        )

        _record_statistics(episode_store=episode_store, run=run)


# ---- Private --------------------------------------------


def _check_rewards(*, episode_store: RewardedEpisodeStore) -> None:
    if np.isnan(episode_store.rewards[episode_store.indices_with_successor]).any():
        raise ValueError("There are some rewards missing.")


def _record_statistics(*, episode_store: RewardedEpisodeStore, run: Run) -> None:
    run.summary["1_episode_count"] = len(episode_store.episode_end_indices)
    run.summary["2_transition_count"] = len(episode_store.states)

    for label, value in summarize_distribution(
        values=episode_store.episode_lengths
    ).items():
        run.summary[f"3_episode_length/{label}"] = value

    for label, value in summarize_distribution(
        values=episode_store.rewards[episode_store.indices_with_successor]
    ).items():
        run.summary[f"4_reward/{label}"] = value

    run.summary["5_state_dimensionality"] = episode_store.states.shape[1]
    run.summary["6_action_dimensionality"] = episode_store.actions.shape[1]


if __name__ == "__main__":
    main()
