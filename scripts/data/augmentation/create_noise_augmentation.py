import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import tyro
import wandb
from numpy.random import Generator
from rich.pretty import pretty_repr

from scripts.ashared import compute_config_hash, create_logger
from scripts.shared.data import (
    DemonstrationData,
    load_demonstration_data_from_artifact,
    write_zarr,
)
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH

logger = create_logger()


@dataclass
class Config:
    use_artifact: str
    # Length of subtrajectories to extract and augment.
    window_length: int
    # Fraction of all candidate subtrajectories to sample and add as noisy episodes (0, 1].
    # E.g. 0.3 means 30% of all sliding-window subtrajectories become new noisy episodes.
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
        project="abc",
        job_type="augmentation-noisy",
        config=asdict(config),
        dir=OUTPUT_DIRECTORY_PATH,
    ) as run:
        artifact = run.use_artifact(artifact_or_name=config.use_artifact)

        demonstration_data = load_demonstration_data_from_artifact(
            artifact=artifact, logger=logger
        )

        logger.info(
            "Extracting all sliding-windows of length %d ...", config.window_length
        )

        augmented_demonstration_data = _augment_with_noisy_episodes(
            demonstration_data=demonstration_data,
            window_length=config.window_length,
            augmentation_ratio=config.augmentation_ratio,
            noise_scale=config.noise_scale,
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


def _augment_with_noisy_episodes(
    demonstration_data: DemonstrationData,
    window_length: int,
    augmentation_ratio: float,
    noise_scale: float,
    rng: Generator,
) -> DemonstrationData:

    noise_std = compute_noise_std(
        samples=demonstration_data.states, noise_scale=noise_scale
    )  # (state_dimensionality,)

    sliding_window_slices = _get_sliding_window_slices(
        episode_end_indices=demonstration_data.episode_end_indices,
        window_length=window_length,
    )

    sampled_window_indices = _sample_random_indices(
        population_size=len(sliding_window_slices),
        sampling_ratio=augmentation_ratio,
        rng=rng,
    )  # (sampled_window_count,)

    augmented_states: list[npt.NDArray[np.float32]] = [
        demonstration_data.states
    ]  # (original_transition_count, state_dimensionality)
    augmented_actions: list[npt.NDArray[np.float32]] = [
        demonstration_data.actions
    ]  # (original_transition_count, action_dimensionality)
    augmented_episode_end_indices: list[int] = []

    # End indices continue from the original data, since the noisy episodes are appended after it.
    cumulative_end_index = int(demonstration_data.episode_end_indices[-1])

    for sampled_window_index in sampled_window_indices:
        window_slice = sliding_window_slices[int(sampled_window_index)]

        window_states = demonstration_data.states[
            window_slice
        ]  # (window_length, state_dimensionality)
        window_actions = demonstration_data.actions[
            window_slice
        ]  # (window_length, action_dimensionality)

        state_noise = rng.normal(
            loc=0.0,
            scale=noise_std,
            size=(window_length, demonstration_data.states.shape[1]),
        ).astype(dtype=window_states.dtype)  # (window_length, state_dimensionality)

        augmented_states.append(window_states + state_noise)
        augmented_actions.append(window_actions.copy())

        cumulative_end_index += window_length
        augmented_episode_end_indices.append(cumulative_end_index)

    return DemonstrationData(
        states=np.concatenate(
            augmented_states, axis=0
        ),  # (original_transition_count + sampled_window_count * window_length, state_dimensionality)
        actions=np.concatenate(
            augmented_actions, axis=0
        ),  # (original_transition_count + sampled_window_count * window_length, action_dimensionality)
        episode_end_indices=np.concatenate(
            [
                demonstration_data.episode_end_indices,
                np.asarray(
                    augmented_episode_end_indices,
                    dtype=demonstration_data.episode_end_indices.dtype,
                ),
            ]
        ),  # (original_episode_count + sampled_window_count,)
    )


def _sample_random_indices(
    population_size: int, sampling_ratio: float, rng: Generator
) -> npt.NDArray[np.int64]:  # (sample_size,)
    sample_size = int(population_size * sampling_ratio)

    if sample_size < 1:
        raise ValueError(
            f"sample_size={sample_size} must be at least 1. "
            f"Got sampling_ratio={sampling_ratio} and population_size={population_size}."
        )

    return rng.choice(a=population_size, size=sample_size, replace=False)


def _get_sliding_window_slices(
    episode_end_indices: npt.NDArray[np.int64],  # (episode_count,)
    window_length: int,
) -> list[slice]:  # len == total_window_count
    episode_start_indices = np.concatenate(
        (np.zeros(shape=1, dtype=episode_end_indices.dtype), episode_end_indices[:-1])
    )  # (episode_count,)

    all_window_slices: list[slice] = []

    skipped_episode_count = 0

    for episode_start_index, episode_end_index in zip(
        episode_start_indices, episode_end_indices
    ):
        episode_length: int = episode_end_index - episode_start_index

        if episode_length < window_length:
            skipped_episode_count += 1
            continue

        window_count = episode_length - window_length + 1

        for window_index_offset in range(window_count):
            window_start_index: int = episode_start_index + window_index_offset
            window_end_index = window_start_index + window_length

            all_window_slices.append(slice(window_start_index, window_end_index))

    if skipped_episode_count > 0:
        logger.info(
            "Skipped %d episodes shorter than subtrajectory_length=%d.",
            skipped_episode_count,
            window_length,
        )

    return all_window_slices


def compute_noise_std(
    samples: npt.NDArray[np.float32],  # (N, dimensionality)
    noise_scale: float,
) -> npt.NDArray[np.float32]:  # (dimensionality,)
    min_per_dimension: npt.NDArray[np.float32] = samples.min(
        axis=0
    )  # (dimensionality,)
    max_per_dimension: npt.NDArray[np.float32] = samples.max(
        axis=0
    )  # (dimensionality,)
    range_per_dimension = max_per_dimension - min_per_dimension  # (dimensionality,)
    return noise_scale * range_per_dimension


if __name__ == "__main__":
    main()
