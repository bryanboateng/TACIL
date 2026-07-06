import numpy as np
import wandb
from wandb import Run

from scripts.data.augmentation.tacil.create_data.episode_assembly import (
    AugmentationResult,
)


def log_dataset_statistics(
    *,
    augmentation_result: AugmentationResult,
    synthetic_part_length: int,
    source_episode_count: int,
    source_transition_count: int,
    run: Run,
) -> None:
    run.summary["candidate_episodes"] = _candidate_episodes_summary(
        augmentation_counts=augmentation_result.counts
    )
    run.summary["added"] = _added_summary(
        augmentation_report=augmentation_result,
        synthetic_part_length=synthetic_part_length,
        source_episode_count=source_episode_count,
        source_transition_count=source_transition_count,
    )
    run.summary["dataset"] = _dataset_summary(
        augmentation_report=augmentation_result,
        synthetic_part_length=synthetic_part_length,
        source_episode_count=source_episode_count,
        source_transition_count=source_transition_count,
    )
    run.summary["bound_threshold"] = augmentation_result.error_bound_threshold
    run.summary["uncertainty_threshold"] = augmentation_result.uncertainty_threshold

    run.log(
        {
            "figures/error_bounds": _clipped_histogram(
                values=augmentation_result.error_bounds
            ),
            "figures/uncertainties": _clipped_histogram(
                values=augmentation_result.uncertainties
            ),
        }
    )


def _clipped_histogram(*, values: np.ndarray) -> wandb.Histogram:  # (step_count,)
    clipped_values = values[values <= float(np.percentile(a=values, q=99.5))]

    # "auto" can pick more bins than wandb allows, so cap the bin count.
    bin_count = min(len(np.histogram_bin_edges(clipped_values, bins="auto")) - 1, 512)
    return wandb.Histogram(np_histogram=np.histogram(a=clipped_values, bins=bin_count))


def _candidate_episodes_summary(
    *, augmentation_counts: AugmentationResult.OutcomeCounts
) -> dict:
    anchor_count = (
        augmentation_counts.accepted
        + augmentation_counts.rejected
        + augmentation_counts.not_converged
    )
    return {
        "anchor_count": anchor_count,
        "counts": {
            "not_converged": augmentation_counts.not_converged,
            "rejected": augmentation_counts.rejected,
            "accepted": augmentation_counts.accepted,
        },
        "fractions": {
            "not_converged": augmentation_counts.not_converged / anchor_count,
            "rejected": augmentation_counts.rejected / anchor_count,
            "accepted": augmentation_counts.accepted / anchor_count,
        },
    }


def _added_summary(
    *,
    augmentation_report: AugmentationResult,
    synthetic_part_length: int,
    source_episode_count: int,
    source_transition_count: int,
) -> dict:
    added_episode_count = len(augmentation_report.episode_store.episode_end_indices)
    added_transition_count = int(augmentation_report.episode_store.states.shape[0])
    added_synthetic_count = augmentation_report.counts.accepted * synthetic_part_length
    added_expert_count = added_transition_count - added_synthetic_count

    return {
        "episodes": {
            "count": added_episode_count,
            "fraction_of_source": added_episode_count / source_episode_count,
        },
        "transitions": {
            "count": added_transition_count,
            "fraction_of_source": added_transition_count / source_transition_count,
        },
        "composition": {
            "count": {"synthetic": added_synthetic_count, "expert": added_expert_count},
            "fraction": {
                "synthetic": added_synthetic_count / added_transition_count,
                "expert": added_expert_count / added_transition_count,
            },
        },
    }


def _dataset_summary(
    *,
    augmentation_report: AugmentationResult,
    synthetic_part_length: int,
    source_episode_count: int,
    source_transition_count: int,
) -> dict:
    added_episode_count = len(augmentation_report.episode_store.episode_end_indices)
    added_transition_count = int(augmentation_report.episode_store.states.shape[0])

    added_synthetic_count = augmentation_report.counts.accepted * synthetic_part_length
    dataset_transition_count = source_transition_count + added_transition_count
    dataset_expert_count = dataset_transition_count - added_synthetic_count

    dataset_episode_count = source_episode_count + added_episode_count

    return {
        "episodes": {
            "count": dataset_episode_count,
            "fraction_of_source": dataset_episode_count / source_episode_count,
        },
        "transitions": {
            "count": dataset_transition_count,
            "fraction_of_source": dataset_transition_count / source_transition_count,
        },
        "composition": {
            "count": {
                "synthetic": added_synthetic_count,
                "expert": dataset_expert_count,
            },
            "fraction": {
                "synthetic": added_synthetic_count / dataset_transition_count,
                "expert": dataset_expert_count / dataset_transition_count,
            },
        },
    }
