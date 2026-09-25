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
    upload_episodes,
)
from scripts.shared.distribution_summary import summarize_distribution
from scripts.shared.episodes import (
    RewardedEpisodes,
    calculate_episode_lengths,
    calculate_indices_with_successor,
)
from scripts.shared.log import create_logger
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH

logger = create_logger()


@dataclass
class Config:
    environment: EnvironmentName
    dataset_name: str


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

    episodes = module.build_episodes()
    _check_rewards(episodes=episodes)

    episode_count = len(episodes.episode_end_indices)

    with wandb.init(
        project=DATASET_PROJECT,
        job_type="expert-data-ingestion",
        dir=OUTPUT_DIRECTORY_PATH,
    ) as run:
        upload_episodes(
            episodes=episodes,
            dataset_name=build_dataset_name(
                stem=config.dataset_name,
                episode_count=episode_count,
                source_episode_count=episode_count,
            ),
            run=run,
        )

        _record_statistics(episodes=episodes, run=run)


# ---- Private --------------------------------------------


def _check_rewards(*, episodes: RewardedEpisodes) -> None:
    indices_with_successor = calculate_indices_with_successor(
        episode_end_indices=episodes.episode_end_indices,
        total_timestep_count=len(episodes.states),
    )  # (transition_count,)
    if np.isnan(episodes.rewards[indices_with_successor]).any():
        raise ValueError("There are some rewards missing.")


def _record_statistics(*, episodes: RewardedEpisodes, run: Run) -> None:
    run.summary["1_episode_count"] = len(episodes.episode_end_indices)
    run.summary["2_transition_count"] = len(episodes.states)

    episode_lengths = calculate_episode_lengths(
        episode_end_indices=episodes.episode_end_indices
    )  # (episode_count,)
    for label, value in summarize_distribution(values=episode_lengths).items():
        run.summary[f"3_episode_length/{label}"] = value

    indices_with_successor = calculate_indices_with_successor(
        episode_end_indices=episodes.episode_end_indices,
        total_timestep_count=len(episodes.states),
    )  # (transition_count,)
    for label, value in summarize_distribution(
        values=episodes.rewards[indices_with_successor]
    ).items():
        run.summary[f"4_reward/{label}"] = value

    run.summary["5_state_dimensionality"] = episodes.states.shape[1]
    run.summary["6_action_dimensionality"] = episodes.actions.shape[1]


if __name__ == "__main__":
    main()
