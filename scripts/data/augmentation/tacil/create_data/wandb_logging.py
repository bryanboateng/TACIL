import tempfile
from pathlib import Path

import numpy as np
import wandb
import zarr
from wandb import Run

from scripts.data.augmentation.tacil.create_data.create_augmented_data import (
    AugmentationResult,
    Counts,
)
from scripts.shared.data import EpisodeDataset


def _log_summary_to_wandb(
    *,
    run: Run,
    augmentation_result: AugmentationResult,
    source_episode_count: int,
    source_transition_count: int,
) -> None:
    run.summary["candidate_episodes"] = _candidate_episodes_summary(
        augmentation_counts=augmentation_result.counts
    )
    run.summary["added"] = _added_summary(
        result=augmentation_result,
        source_episode_count=source_episode_count,
        source_transition_count=source_transition_count,
    )
    run.summary["dataset"] = _dataset_summary(
        result=augmentation_result,
        source_episode_count=source_episode_count,
        source_transition_count=source_transition_count,
    )
    run.summary["bound_threshold"] = augmentation_result.bound_threshold

    bounds = augmentation_result.bounds
    clipped_bounds = bounds[bounds <= float(np.percentile(bounds, q=99.5))]
    run.log(
        {
            "figures/bounds": wandb.Histogram(
                np_histogram=np.histogram(clipped_bounds, bins="auto")
            )
        }
    )


def _fraction(*, part: int, whole: int) -> float:
    return part / whole if whole else 0.0


def _candidate_episodes_summary(*, augmentation_counts: Counts) -> dict:
    anchor_count = (
        augmentation_counts.accepted
        + augmentation_counts.rejected
        + augmentation_counts.not_converged
    )
    return {
        "anchor_count": anchor_count,
        "count": {
            "not_converged": augmentation_counts.not_converged,
            "rejected": augmentation_counts.rejected,
            "accepted": augmentation_counts.accepted,
        },
        "fraction": {
            "not_converged": _fraction(
                part=augmentation_counts.not_converged, whole=anchor_count
            ),
            "rejected": _fraction(
                part=augmentation_counts.rejected, whole=anchor_count
            ),
            "accepted": _fraction(
                part=augmentation_counts.accepted, whole=anchor_count
            ),
        },
    }


def _added_summary(
    *,
    result: AugmentationResult,
    source_episode_count: int,
    source_transition_count: int,
) -> dict:
    """Accepted episodes that enter the augmented dataset.

    Each accepted anchor yields exactly one episode, so added episodes ==
    accepted episodes. `ratio_of_original` is the added size divided by the
    original size and may exceed 1. `by_source` partitions the added
    transitions (a fixed synthetic prefix per episode, the rest verbatim
    expert copies), so its two fractions sum to 1.
    """
    added_episode_count = len(result.episode_end_indices)
    added_transition_count = int(result.states.shape[0])
    added_synthetic_count = result.counts.accepted * result.synthetic_part_length
    added_expert_count = added_transition_count - added_synthetic_count

    return {
        "episodes": {
            "count": added_episode_count,
            "ratio_of_original": _fraction(
                part=added_episode_count, whole=source_episode_count
            ),
        },
        "transitions": {
            "count": added_transition_count,
            "ratio_of_original": _fraction(
                part=added_transition_count, whole=source_transition_count
            ),
            "by_source": {
                "count": {
                    "synthetic": added_synthetic_count,
                    "expert": added_expert_count,
                },
                "fraction": {
                    "synthetic": _fraction(
                        part=added_synthetic_count, whole=added_transition_count
                    ),
                    "expert": _fraction(
                        part=added_expert_count, whole=added_transition_count
                    ),
                },
            },
        },
    }


def _dataset_summary(
    *,
    result: AugmentationResult,
    source_episode_count: int,
    source_transition_count: int,
) -> dict:
    added_episode_count = len(result.episode_dataset.episode_end_indices)
    added_transition_count = int(result.episode_dataset.states.shape[0])

    added_synthetic_count = result.counts.accepted * result.synthetic_part_length
    dataset_transition_count = source_transition_count + added_transition_count
    dataset_expert_count = dataset_transition_count - added_synthetic_count

    return {
        "episodes": {"count": source_episode_count + added_episode_count},
        "transitions": {
            "count": dataset_transition_count,
            "by_source": {
                "count": {
                    "synthetic": added_synthetic_count,
                    "expert": dataset_expert_count,
                },
                "fraction": {
                    "synthetic": _fraction(
                        part=added_synthetic_count, whole=dataset_transition_count
                    ),
                    "expert": _fraction(
                        part=dataset_expert_count, whole=dataset_transition_count
                    ),
                },
            },
        },
    }


def _log_augmentation_to_wandb(
    *,
    run: Run,
    expert_episode_dataset: EpisodeDataset,
    augmentation_episode_dataset: EpisodeDataset,
) -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_directory_path = Path(temporary_directory)
        combined_zarr_path = temporary_directory_path / "combined.zarr"

        _write_combined_zarr(
            output_path=combined_zarr_path,
            expert_episode_dataset=expert_episode_dataset,
            augmentation_episode_dataset=augmentation_episode_dataset,
        )

        dataset_artifact = wandb.Artifact(
            name=f"tacil-dataset-{run.name}", type="dataset"
        )
        dataset_artifact.add_dir(str(combined_zarr_path))
        run.log_artifact(dataset_artifact).wait()


def _write_combined_zarr(
    *,
    output_path: Path,
    expert_episode_dataset: EpisodeDataset,
    augmentation_episode_dataset: EpisodeDataset,
) -> None:
    combined_data = combine_episode_datasets(
        expert_episode_dataset=expert_episode_dataset,
        augmentation_episode_dataset=augmentation_episode_dataset,
    )

    write_episode_dataset(output_path=output_path, episode_dataset=combined_data)


def combine_episode_datasets(
    *,
    expert_episode_dataset: EpisodeDataset,
    augmentation_episode_dataset: EpisodeDataset,
) -> EpisodeDataset:
    states = np.concatenate(
        [expert_episode_dataset.states, augmentation_episode_dataset.states], axis=0
    )

    actions = np.concatenate(
        [expert_episode_dataset.actions, augmentation_episode_dataset.actions], axis=0
    )

    original_timestep_count = expert_episode_dataset.states.shape[0]
    augmented_episode_ends_absolute = (
        original_timestep_count + augmentation_episode_dataset.episode_end_indices
    )

    episode_end_indices = np.concatenate(
        [expert_episode_dataset.episode_end_indices, augmented_episode_ends_absolute],
        axis=0,
    )

    return EpisodeDataset(
        states=states, actions=actions, episode_end_indices=episode_end_indices
    )


def write_episode_dataset(
    *, output_path: Path, episode_dataset: EpisodeDataset
) -> None:
    root = zarr.open_group(str(output_path), mode="w", zarr_format=2)

    data_group = root.create_group("data")
    data_group.create_array("state", data=episode_dataset.states)
    data_group.create_array("action", data=episode_dataset.actions)

    meta_group = root.create_group("meta")
    meta_group.create_array("episode_ends", data=episode_dataset.episode_end_indices)
