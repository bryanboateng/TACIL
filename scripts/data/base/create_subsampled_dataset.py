import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tyro
import wandb
from rich.pretty import pretty_repr
from wandb import Run

from scripts.ashared import create_logger
from scripts.shared.data import (
    EpisodeDataset,
    load_episode_dataset,
    save_episode_dataset,
    upload_episode_dataset,
)
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH

logger = create_logger()


@dataclass
class Config:
    fraction: float
    seed: int = 42

    def __post_init__(self):
        if not 0.0 <= self.fraction <= 1.0:
            raise ValueError(f"fraction must be in [0, 1], got {self.fraction}")


def main():
    config = tyro.cli(f=Config, config=[tyro.conf.UsePythonSyntaxForLiteralCollections])
    logger.info(pretty_repr(config))
    _run(config=config)


def _run(*, config: Config):
    rng = np.random.default_rng(seed=config.seed)

    with wandb.init(
        project="abc",
        job_type="create-subsampled-dataset",
        config=asdict(config),
        dir=OUTPUT_DIRECTORY_PATH,
    ) as run:
        expert_episode_dataset = load_episode_dataset(
            episode_file_path=Path(
                run.use_artifact(artifact_or_name="expert-dataset:latest").download()
            )
        )

        subsampled_episode_dataset = _subsample_episodes(
            episode_dataset=expert_episode_dataset, fraction=config.fraction, rng=rng
        )

        _log_dataset_statistics(
            expert_episode_dataset=expert_episode_dataset,
            subsampled_episode_dataset=subsampled_episode_dataset,
            run=run,
        )

        upload_episode_dataset(
            episode_dataset=subsampled_episode_dataset,
            name="subsampled-dataset",
            run=run,
        )


def _subsample_episodes(
    *, episode_dataset: EpisodeDataset, fraction: float, rng: np.random.Generator
) -> EpisodeDataset:

    episode_count = len(episode_dataset.episode_end_indices)

    selected_episode_indices = np.sort(
        a=rng.choice(a=episode_count, size=int(episode_count * fraction), replace=False)
    )  # (selected_episode_count,)

    episode_start_indices = np.concatenate(
        (
            np.zeros(shape=1, dtype=episode_dataset.episode_end_indices.dtype),
            episode_dataset.episode_end_indices[:-1],
        )
    )  # (episode_count,)

    selected_transition_indices_list = []

    subsampled_episode_end_indices = []

    cumulative_transition_count = 0

    for selected_episode_index in selected_episode_indices:
        selected_episode_index: int
        start_index: int = episode_start_indices[selected_episode_index]
        end_index: int = episode_dataset.episode_end_indices[selected_episode_index]

        episode_transition_indices = np.arange(
            start_index, stop=end_index
        )  # (episode_length,)

        selected_transition_indices_list.append(episode_transition_indices)

        cumulative_transition_count += len(episode_transition_indices)
        subsampled_episode_end_indices.append(cumulative_transition_count)

    selected_transition_indices_np_array = np.concatenate(
        selected_transition_indices_list
    )  # (selected_transition_count,)

    return EpisodeDataset(
        states=episode_dataset.states[
            selected_transition_indices_np_array
        ],  # (selected_transition_count, state_dimensionality)
        actions=episode_dataset.actions[
            selected_transition_indices_np_array
        ],  # (selected_transition_count, action_dimensionality)
        episode_end_indices=np.asarray(
            subsampled_episode_end_indices
        ),  # (selected_episode_count,)
    )


def _log_dataset_statistics(
    expert_episode_dataset: EpisodeDataset,
    subsampled_episode_dataset: EpisodeDataset,
    run: Run,
):
    expert_episode_count = len(expert_episode_dataset.episode_end_indices)
    expert_transition_count = len(expert_episode_dataset.states)

    run.summary["source_dataset"] = {
        "episode_count": expert_episode_count,
        "transitions_count": expert_transition_count,
    }

    selected_episode_count = len(subsampled_episode_dataset.episode_end_indices)
    selected_transition_count = len(subsampled_episode_dataset.states)

    run.summary["subsampled_dataset"] = {
        "episodes": {
            "count": selected_episode_count,
            "fraction_of_source": selected_episode_count / expert_episode_count,
        },
        "transitions": {
            "count": selected_transition_count,
            "fraction_of_source": (selected_transition_count / expert_transition_count),
        },
    }


if __name__ == "__main__":
    main()
