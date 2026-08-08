from dataclasses import asdict, dataclass

import numpy as np
import tyro
import wandb
from rich.pretty import pretty_repr
from wandb import Run

from scripts.shared.data import (
    EpisodeStore,
    download_episode_store,
    upload_episode_store,
)
from scripts.shared.log import create_logger
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH
from scripts.shared.wandb_config import WandbConfig

logger = create_logger()


@dataclass
class Config:
    wandb: WandbConfig

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
        project=config.wandb.project,
        notes=config.wandb.notes,
        job_type="create-subsampled-dataset",
        config=asdict(config),
        dir=OUTPUT_DIRECTORY_PATH,
    ) as run:
        expert_episode_store = download_episode_store(
            artifact_name="expert-dataset:latest", run=run
        )

        subsampled_episode_store = _subsample_episodes(
            episode_store=expert_episode_store, fraction=config.fraction, rng=rng
        )

        _log_dataset_statistics(
            expert_episode_store=expert_episode_store,
            subsampled_episode_store=subsampled_episode_store,
            run=run,
        )

        upload_episode_store(
            episode_store=subsampled_episode_store, name="subsampled-dataset", run=run
        )


def _subsample_episodes(
    *, episode_store: EpisodeStore, fraction: float, rng: np.random.Generator
) -> EpisodeStore:

    episode_count = len(episode_store.episode_end_indices)

    selected_episode_indices = np.sort(
        a=rng.choice(a=episode_count, size=int(episode_count * fraction), replace=False)
    )  # (selected_episode_count,)

    episode_start_indices = np.concatenate(
        (
            np.zeros(shape=1, dtype=episode_store.episode_end_indices.dtype),
            episode_store.episode_end_indices[:-1],
        )
    )  # (episode_count,)

    selected_transition_indices_list = []

    subsampled_episode_end_indices = []

    cumulative_transition_count = 0

    for selected_episode_index in selected_episode_indices:
        selected_episode_index: int
        start_index: int = episode_start_indices[selected_episode_index]
        end_index: int = episode_store.episode_end_indices[selected_episode_index]

        episode_transition_indices = np.arange(
            start_index, stop=end_index
        )  # (episode_length,)

        selected_transition_indices_list.append(episode_transition_indices)

        cumulative_transition_count += len(episode_transition_indices)
        subsampled_episode_end_indices.append(cumulative_transition_count)

    selected_transition_indices_np_array = np.concatenate(
        selected_transition_indices_list
    )  # (selected_transition_count,)

    return EpisodeStore(
        states=episode_store.states[
            selected_transition_indices_np_array
        ],  # (selected_transition_count, state_dimensionality)
        actions=episode_store.actions[
            selected_transition_indices_np_array
        ],  # (selected_transition_count, action_dimensionality)
        episode_end_indices=np.asarray(
            subsampled_episode_end_indices
        ),  # (selected_episode_count,)
    )


def _log_dataset_statistics(
    expert_episode_store: EpisodeStore, subsampled_episode_store: EpisodeStore, run: Run
):
    expert_episode_count = len(expert_episode_store.episode_end_indices)
    expert_transition_count = len(expert_episode_store.states)

    run.summary["source_dataset"] = {
        "episode_count": expert_episode_count,
        "transitions_count": expert_transition_count,
    }

    selected_episode_count = len(subsampled_episode_store.episode_end_indices)
    selected_transition_count = len(subsampled_episode_store.states)

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
