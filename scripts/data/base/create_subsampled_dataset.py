import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tyro
import wandb
from rich.pretty import pretty_repr

from scripts.ashared import create_logger
from scripts.shared.data import (
    DemonstrationData,
    load_demonstration_data_from_artifact,
    save_episode_dataset,
)
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH

logger = create_logger()


@dataclass
class Config:
    fraction: float
    seed: int = 42


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
        artifact = run.use_artifact(artifact_or_name="expert-dataset:latest")

        demonstration_data = load_demonstration_data_from_artifact(
            artifact=artifact, logger=logger
        )

        logger.info("Subsampling dataset with %g%% ...", config.fraction * 100)
        subsampled_episodes = _subsample_episodes(
            demonstration_data=demonstration_data, fraction=config.fraction, rng=rng
        )

        logger.info("Subsampling dataset completed.")
        logger.info("# Transitions: %d", len(subsampled_episodes.states))
        logger.info("# Episodes: %d", len(subsampled_episodes.episode_end_indices))

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_zarr_path = Path(temporary_directory) / "dataset.zarr"

            save_episode_dataset(episode_dataset=, save_file_path=output_zarr_path)

            new_artifact = wandb.Artifact(
                name=f"subsampled-dataset-{run.name}", type="dataset"
            )

            new_artifact.add_dir(str(output_zarr_path))

            run.log_artifact(new_artifact)


def _subsample_episodes(
    *, demonstration_data: DemonstrationData, fraction: float, rng: np.random.Generator
) -> DemonstrationData:
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    episode_count = len(demonstration_data.episode_end_indices)

    selected_episodes_count = int(episode_count * fraction)

    if selected_episodes_count < 1:
        raise ValueError(
            f"fraction={fraction} results in selecting 0 episodes "
            f"from episode_count={episode_count}. "
            f"Increase the fraction."
        )

    selected_episode_indices = np.sort(
        a=rng.choice(a=episode_count, size=selected_episodes_count, replace=False)
    )

    episode_start_indices = np.concatenate(
        (
            np.zeros(shape=1, dtype=demonstration_data.episode_end_indices.dtype),
            demonstration_data.episode_end_indices[:-1],
        )
    )

    selected_transition_indices_list = []

    subsampled_episode_end_indices = []

    cumulative_transition_count = 0

    for selected_episode_index in selected_episode_indices:
        selected_episode_index: int
        start_index: int = episode_start_indices[selected_episode_index]
        end_index: int = demonstration_data.episode_end_indices[selected_episode_index]

        episode_transition_indices = np.arange(start_index, stop=end_index)

        selected_transition_indices_list.append(episode_transition_indices)

        cumulative_transition_count += len(episode_transition_indices)
        subsampled_episode_end_indices.append(cumulative_transition_count)

    selected_transition_indices_np_array = np.concatenate(
        selected_transition_indices_list
    )

    return DemonstrationData(
        states=demonstration_data.states[selected_transition_indices_np_array],
        actions=demonstration_data.actions[selected_transition_indices_np_array],
        episode_end_indices=np.asarray(subsampled_episode_end_indices),
    )


if __name__ == "__main__":
    main()
