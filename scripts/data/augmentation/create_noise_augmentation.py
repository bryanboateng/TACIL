from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import tyro
import wandb
from numpy.random import Generator
from rich.pretty import pretty_repr
from wandb import Run

from scripts.shared.data import EpisodeStore, load_episode_store, upload_episode_store
from scripts.shared.log import create_logger
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH

logger = create_logger()


@dataclass
class Config:
    wandb_project: str

    dataset_artifact: str
    # Fraction of all candidate subtrajectories to sample and add as noisy episodes (0, 1].
    # E.g. 0.3 means 30% of all sliding-window subtrajectories become new noisy episodes.
    augmentation_ratio: float
    # Length of subtrajectories to extract and augment.
    window_length: int = 21
    # Noise std as a fraction of each state dimension's observed min-max range.
    # E.g. 0.02 = 2% of range per dimension.
    noise_fraction: float = 0.05
    seed: int = 42


def main():
    config = tyro.cli(f=Config, config=[tyro.conf.UsePythonSyntaxForLiteralCollections])
    logger.info(pretty_repr(config))
    _run(config=config)


def _run(*, config: Config):
    rng = np.random.default_rng(seed=config.seed)

    with wandb.init(
        project=config.wandb_project,
        job_type="create-noisy-dataset",
        config=asdict(config),
        dir=OUTPUT_DIRECTORY_PATH,
    ) as run:
        source_episode_store = load_episode_store(
            episode_file_path=Path(
                run.use_artifact(artifact_or_name=config.dataset_artifact).download()
            )
        )

        augmented_episode_store = _augment_with_noisy_episodes(
            episode_store=source_episode_store,
            window_length=config.window_length,
            augmentation_ratio=config.augmentation_ratio,
            noise_fraction=config.noise_fraction,
            rng=rng,
        )

        _log_dataset_statistics(
            source_episode_store=source_episode_store,
            augmented_episode_store=augmented_episode_store,
            run=run,
        )

        upload_episode_store(
            episode_store=augmented_episode_store, name="noisy-dataset", run=run
        )


def _augment_with_noisy_episodes(
    *,
    episode_store: EpisodeStore,
    window_length: int,
    augmentation_ratio: float,
    noise_fraction: float,
    rng: Generator,
) -> EpisodeStore:

    noise_std = compute_noise_std(
        samples=episode_store.states, noise_fraction=noise_fraction
    )  # (state_dimensionality,)

    sliding_window_slices = _get_sliding_window_slices(
        episode_end_indices=episode_store.episode_end_indices,
        window_length=window_length,
    )

    sampled_window_indices = _sample_random_indices(
        population_size=len(sliding_window_slices),
        sampling_ratio=augmentation_ratio,
        rng=rng,
    )  # (sampled_window_count,)

    augmented_states: list[npt.NDArray[np.float32]] = [
        episode_store.states
    ]  # (original_transition_count, state_dimensionality)
    augmented_actions: list[npt.NDArray[np.float32]] = [
        episode_store.actions
    ]  # (original_transition_count, action_dimensionality)
    augmented_episode_end_indices: list[int] = []

    augmented_artificial_end_flags: list[bool] = []
    source_episode_end_index_set = set(episode_store.episode_end_indices.tolist())

    # End indices continue from the original data, since the noisy episodes are appended after it.
    cumulative_end_index = int(episode_store.episode_end_indices[-1])

    for sampled_window_index in sampled_window_indices:
        window_slice = sliding_window_slices[int(sampled_window_index)]
        augmented_artificial_end_flags.append(
            int(window_slice.stop) not in source_episode_end_index_set
        )

        window_states = episode_store.states[
            window_slice
        ]  # (window_length, state_dimensionality)
        window_actions = episode_store.actions[
            window_slice
        ]  # (window_length, action_dimensionality)

        state_noise = rng.normal(
            loc=0.0,
            scale=noise_std,
            size=(window_length, episode_store.states.shape[1]),
        ).astype(dtype=window_states.dtype)  # (window_length, state_dimensionality)

        augmented_states.append(window_states + state_noise)
        augmented_actions.append(window_actions.copy())

        cumulative_end_index += window_length
        augmented_episode_end_indices.append(cumulative_end_index)

    return EpisodeStore(
        states=np.concatenate(
            augmented_states, axis=0
        ),  # (original_transition_count + sampled_window_count * window_length, state_dimensionality)
        actions=np.concatenate(
            augmented_actions, axis=0
        ),  # (original_transition_count + sampled_window_count * window_length, action_dimensionality)
        episode_end_indices=np.concatenate(
            [
                episode_store.episode_end_indices,
                np.asarray(
                    augmented_episode_end_indices,
                    dtype=episode_store.episode_end_indices.dtype,
                ),
            ]
        ),  # (original_episode_count + sampled_window_count,)
        artificial_end_mask=np.concatenate(
            [
                (
                    episode_store.artificial_end_mask
                    if episode_store.artificial_end_mask is not None
                    else np.zeros(
                        shape=len(episode_store.episode_end_indices), dtype=np.bool_
                    )
                ),
                np.asarray(augmented_artificial_end_flags, dtype=np.bool_),
            ]
        ),  # (original_episode_count + sampled_window_count,)
    )


def _sample_random_indices(
    *, population_size: int, sampling_ratio: float, rng: Generator
) -> npt.NDArray[np.int64]:  # (sample_size,)
    sample_size = int(population_size * sampling_ratio)

    if sample_size < 1:
        raise ValueError(
            f"sample_size={sample_size} must be at least 1. "
            f"Got sampling_ratio={sampling_ratio} and population_size={population_size}."
        )

    return rng.choice(a=population_size, size=sample_size, replace=False)


def _get_sliding_window_slices(
    *,
    episode_end_indices: npt.NDArray[np.int64],  # (episode_count,)
    window_length: int,
) -> list[slice]:  # len == total_window_count
    episode_start_indices = np.concatenate(
        (np.zeros(shape=1, dtype=episode_end_indices.dtype), episode_end_indices[:-1])
    )  # (episode_count,)

    all_window_slices: list[slice] = []

    for episode_start_index, episode_end_index in zip(
        episode_start_indices, episode_end_indices
    ):
        episode_length: int = episode_end_index - episode_start_index

        if episode_length < window_length:
            continue

        window_count = episode_length - window_length + 1

        for window_index_offset in range(window_count):
            window_start_index: int = episode_start_index + window_index_offset
            window_end_index = window_start_index + window_length

            all_window_slices.append(slice(window_start_index, window_end_index))

    return all_window_slices


def compute_noise_std(
    *,
    samples: npt.NDArray[np.float32],  # (N, dimensionality)
    noise_fraction: float,
) -> npt.NDArray[np.float32]:  # (dimensionality,)
    min_per_dimension: npt.NDArray[np.float32] = samples.min(
        axis=0
    )  # (dimensionality,)
    max_per_dimension: npt.NDArray[np.float32] = samples.max(
        axis=0
    )  # (dimensionality,)
    range_per_dimension = max_per_dimension - min_per_dimension  # (dimensionality,)
    return noise_fraction * range_per_dimension


def _log_dataset_statistics(
    source_episode_store: EpisodeStore, augmented_episode_store: EpisodeStore, run: Run
):
    source_episode_count = len(source_episode_store.episode_end_indices)
    source_transition_count = len(source_episode_store.states)

    run.summary["source_dataset"] = {
        "episode_count": source_episode_count,
        "transition_count": source_transition_count,
    }

    new_episode_count = (
        len(augmented_episode_store.episode_end_indices) - source_episode_count
    )
    new_transition_count = len(augmented_episode_store.states) - source_transition_count

    run.summary["added"] = {
        "episodes": {
            "count": new_episode_count,
            "fraction_of_source": new_episode_count / source_episode_count,
        },
        "transitions": {
            "count": new_transition_count,
            "fraction_of_source": new_transition_count / source_transition_count,
        },
    }

    dataset_episode_count = len(augmented_episode_store.episode_end_indices)
    dataset_transition_count = len(augmented_episode_store.states)

    run.summary["augmented_dataset"] = {
        "episodes": {
            "count": dataset_episode_count,
            "fraction_of_source": dataset_episode_count / source_episode_count,
        },
        "transitions": {
            "count": dataset_transition_count,
            "fraction_of_source": dataset_transition_count / source_transition_count,
        },
    }


if __name__ == "__main__":
    main()
