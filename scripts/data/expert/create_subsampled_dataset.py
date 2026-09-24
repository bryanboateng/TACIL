from dataclasses import asdict, dataclass

import numpy as np
import tyro
import wandb
from rich.pretty import pretty_repr
from wandb import Run

from scripts.shared.dataset_artifact import (
    DATASET_PROJECT,
    build_dataset_name,
    download_episodes,
    read_environment_name,
    upload_episodes,
)
from scripts.shared.episodes import Episodes
from scripts.shared.log import create_logger
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH

logger = create_logger()


@dataclass
class Config:
    source_dataset_artifact: str
    episode_count: int

    def __post_init__(self):
        if self.episode_count < 1:
            raise ValueError(
                f"episode_count must be positive, but it is {self.episode_count}."
            )


def main():
    config = tyro.cli(f=Config)
    logger.info(pretty_repr(config))
    _run(config=config)


def _run(*, config: Config):
    with wandb.init(
        project=DATASET_PROJECT,
        job_type="dataset-subsampling",
        config=asdict(config),
        dir=OUTPUT_DIRECTORY_PATH,
    ) as run:
        expert_episodes = download_episodes(
            artifact_name=config.source_dataset_artifact, run=run
        )

        source_episode_count = len(expert_episodes.episode_end_indices)
        source_name = config.source_dataset_artifact.split(":", maxsplit=1)[0]

        if not source_name.endswith(f"-of-{source_episode_count}"):
            raise ValueError(
                f"Subsampling needs a full dataset, but {source_name!r} holds "
                f"{source_episode_count} episodes."
            )

        subsampled_episodes = _subsample_episodes(
            episodes=expert_episodes, episode_count=config.episode_count
        )

        _log_dataset_statistics(
            expert_episodes=expert_episodes,
            subsampled_episodes=subsampled_episodes,
            run=run,
        )

        upload_episodes(
            episodes=subsampled_episodes,
            dataset_name=build_dataset_name(
                environment=read_environment_name(dataset_name=source_name),
                episode_count=len(subsampled_episodes.episode_end_indices),
                source_episode_count=source_episode_count,
            ),
            run=run,
        )


def _subsample_episodes(*, episodes: Episodes, episode_count: int) -> Episodes:

    source_episode_count = len(episodes.episode_end_indices)

    selected_episode_indices = np.sort(
        a=np.random.default_rng(seed=42).permutation(x=source_episode_count)[
            :episode_count
        ]
    )  # (selected_episode_count,)

    # (episode_count,)
    selected_transition_indices_list = []
    subsampled_episode_end_indices = []
    cumulative_transition_count = 0

    for selected_episode_index in selected_episode_indices.tolist():
        start_index: int = episodes.episode_start_indices[selected_episode_index]
        end_index: int = episodes.episode_end_indices[selected_episode_index]

        episode_transition_indices = np.arange(
            start_index, stop=end_index
        )  # (episode_length,)

        selected_transition_indices_list.append(episode_transition_indices)

        cumulative_transition_count += len(episode_transition_indices)
        subsampled_episode_end_indices.append(cumulative_transition_count)

    selected_transition_indices_np_array = np.concatenate(
        selected_transition_indices_list
    )  # (selected_transition_count,)

    return Episodes(
        states=episodes.states[
            selected_transition_indices_np_array
        ],  # (selected_transition_count, state_dimensionality)
        actions=episodes.actions[
            selected_transition_indices_np_array
        ],  # (selected_transition_count, action_dimensionality)
        rewards=episodes.rewards[
            selected_transition_indices_np_array
        ],  # (selected_transition_count,)
        episode_end_indices=np.asarray(
            a=subsampled_episode_end_indices, dtype=np.int64
        ),  # (selected_episode_count,)
    )


def _log_dataset_statistics(
    expert_episodes: Episodes, subsampled_episodes: Episodes, run: Run
):
    expert_episode_count = len(expert_episodes.episode_end_indices)
    expert_transition_count = len(expert_episodes.states)

    run.summary["1_source_dataset/1_episode_count"] = expert_episode_count
    run.summary["1_source_dataset/2_transition_count"] = expert_transition_count

    selected_episode_count = len(subsampled_episodes.episode_end_indices)
    selected_transition_count = len(subsampled_episodes.states)

    run.summary["2_subsampled_dataset/1_episode_count"] = selected_episode_count
    run.summary["2_subsampled_dataset/3_episode_fraction_of_source"] = (
        selected_episode_count / expert_episode_count
    )
    run.summary["2_subsampled_dataset/2_transition_count"] = selected_transition_count
    run.summary["2_subsampled_dataset/4_transition_fraction_of_source"] = (
        selected_transition_count / expert_transition_count
    )


if __name__ == "__main__":
    main()
