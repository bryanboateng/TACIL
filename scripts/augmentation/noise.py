import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import tyro
import wandb
import zarr
from rich.pretty import pretty_repr
from zarr import Array, Group

from scripts.ashared import compute_config_hash, create_logger
from scripts.shared.data import (
    DemonstrationData,
    load_demonstration_data_from_artifact,
    write_zarr,
)

logger = create_logger()


@dataclass
class Config:
    use_artifact: str
    # Ratio of original episodes to augment (0, 1].
    # E.g. 0.5 means 50% of episodes get a noisy copy added → dataset grows by 50%.
    augmentation_ratio: float
    # Noise std as a fraction of each state dimension's observed min-max range.
    # E.g. 0.02 = 2% of range per dimension.
    noise_scale: float = 0.02
    seed: int = 42


def main():
    config = tyro.cli(f=Config, config=[tyro.conf.UsePythonSyntaxForLiteralCollections])
    logger.info(pretty_repr(config))
    _run(config=config)


def _run(config: Config):
    rng = np.random.default_rng(seed=config.seed)

    with wandb.init(
        project="abc", job_type="augmentation-noisy", config=asdict(config)
    ) as run:
        artifact = run.use_artifact(artifact_or_name=config.use_artifact)

        demonstration_data = load_demonstration_data_from_artifact(
            artifact=artifact, logger=logger
        )

        logger.info(
            "Augmenting %g%% of episodes with noise (noise_scale=%.4f) ...",
            config.augmentation_ratio * 100,
            config.noise_scale,
        )

        augmented_demonstration_data = _augment_with_noise(
            demonstration_data=demonstration_data,
            fraction=config.augmentation_ratio,
            noise_fraction=config.noise_scale,
            rng=rng,
        )

        logger.info("Noise augmentation completed.")
        logger.info("# Transitions: %d", len(augmented_demonstration_data.states))
        logger.info(
            "# Episodes: %d", len(augmented_demonstration_data.episode_end_indices)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_zarr_path = Path(tmpdir) / "dataset.zarr"

            write_zarr(
                zarr_file_path=output_zarr_path,
                demonstration_data=augmented_demonstration_data,
            )

            new_artifact = wandb.Artifact(
                name=f"noisy-dataset-{compute_config_hash(config=asdict(config))}",
                type="dataset",
            )

            new_artifact.add_dir(str(output_zarr_path))

            run.log_artifact(new_artifact)


def _augment_with_noise(
    demonstration_data: DemonstrationData,
    fraction: float,
    noise_fraction: float,
    rng: np.random.Generator,
) -> DemonstrationData:
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"augmentation_ratio must be in (0, 1], got {fraction}")

    episode_count = len(demonstration_data.episode_end_indices)

    selected_episode_indices = int(episode_count * fraction)

    if selected_episode_indices < 1:
        raise ValueError(
            f"fraction={fraction} results in selecting 0 episodes "
            f"from episode_count={episode_count}. "
            f"Increase the fraction."
        )

    selected_episode_indices = np.sort(
        rng.choice(a=episode_count, size=selected_episode_indices, replace=False)
    )

    episode_start_indices = np.concatenate(
        (
            np.zeros(shape=1, dtype=demonstration_data.episode_end_indices.dtype),
            demonstration_data.episode_end_indices[:-1],
        )
    )

    noisy_states_list: list[npt.NDArray[np.float32]] = []
    noisy_actions_list: list[npt.NDArray[np.float32]] = []
    noisy_episode_end_indices: list[int] = []

    cumulative_transition_count = int(demonstration_data.episode_end_indices[-1])

    noise_std = _compute_noise_std(
        states=demonstration_data.states, noise_fraction=noise_fraction
    )
    for selected_episode_index in selected_episode_indices:
        start_index: int = episode_start_indices[selected_episode_index]
        end_index: int = demonstration_data.episode_end_indices[selected_episode_index]

        episode_length = end_index - start_index

        noise = rng.normal(
            loc=0.0,
            scale=noise_std,
            size=(episode_length, demonstration_data.states.shape[1]),
        ).astype(np.float32)

        noisy_states_list.append(
            demonstration_data.states[start_index:end_index] + noise
        )
        noisy_actions_list.append(
            demonstration_data.actions[start_index:end_index].copy()
        )

        cumulative_transition_count += episode_length
        noisy_episode_end_indices.append(cumulative_transition_count)

    return DemonstrationData(
        states=np.concatenate([demonstration_data.states] + noisy_states_list, axis=0),
        actions=np.concatenate(
            [demonstration_data.actions] + noisy_actions_list, axis=0
        ),
        episode_end_indices=np.concatenate(
            [
                demonstration_data.episode_end_indices,
                np.asarray(
                    noisy_episode_end_indices,
                    dtype=demonstration_data.episode_end_indices.dtype,
                ),
            ]
        ),
    )


def _compute_noise_std(
    states: npt.NDArray[np.float32], noise_fraction: float
) -> npt.NDArray[np.float32]:

    min_per_dimension: npt.NDArray[np.float32] = states.min(axis=0)
    max_per_dimension: npt.NDArray[np.float32] = states.max(axis=0)

    range_per_dimension = max_per_dimension - min_per_dimension

    return noise_fraction * range_per_dimension


if __name__ == "__main__":
    main()
