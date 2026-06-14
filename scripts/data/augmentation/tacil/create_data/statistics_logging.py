import numpy as np
import wandb
from wandb import Run

from scripts.data.augmentation.tacil.create_data.episode_assembly import (
    AugmentationResult,
)


def log_dataset_statistics(
    *,
    augmentation_result: AugmentationResult,
    source_episode_count: int,
    source_transition_count: int,
    run: Run,
) -> None:
    run.summary["candidate_episodes"] = _candidate_episodes_summary(
        augmentation_counts=augmentation_result.counts
    )
    run.summary["added"] = _added_summary(
        augmentation_report=augmentation_result,
        source_episode_count=source_episode_count,
        source_transition_count=source_transition_count,
    )
    run.summary["dataset"] = _dataset_summary(
        augmentation_report=augmentation_result,
        source_episode_count=source_episode_count,
        source_transition_count=source_transition_count,
    )
    run.summary["bound_threshold"] = augmentation_result.bound_threshold

    bounds = augmentation_result.bounds  # (converged_step_count,)
    clipped_bounds = bounds[
        bounds <= float(np.percentile(bounds, q=99.5))
    ]  # (clipped_step_count,)
    run.log(
        {
            "figures/bounds": wandb.Histogram(
                np_histogram=np.histogram(clipped_bounds, bins="auto")
            )
        }
    )


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
        "count": {
            "not_converged": augmentation_counts.not_converged,
            "rejected": augmentation_counts.rejected,
            "accepted": augmentation_counts.accepted,
        },
        "fraction": {
            "not_converged": augmentation_counts.not_converged / anchor_count,
            "rejected": augmentation_counts.rejected / anchor_count,
            "accepted": augmentation_counts.accepted / anchor_count,
        },
    }


def _added_summary(
    *,
    augmentation_report: AugmentationResult,
    source_episode_count: int,
    source_transition_count: int,
) -> dict:
    added_episode_count = len(augmentation_report.episode_dataset.episode_end_indices)
    added_transition_count = int(augmentation_report.episode_dataset.states.shape[0])
    added_synthetic_count = (
        augmentation_report.counts.accepted * augmentation_report.synthetic_part_length
    )
    added_expert_count = added_transition_count - added_synthetic_count

    return {
        "episodes": {
            "count": added_episode_count,
            "fraction_of_source": added_episode_count / source_episode_count,
        },
        "transitions": {
            "count": added_transition_count,
            "fraction_of_source": added_transition_count / source_transition_count,
            "by_source": {
                "count": {
                    "synthetic": added_synthetic_count,
                    "expert": added_expert_count,
                },
                "fraction": {
                    "synthetic": added_synthetic_count / added_transition_count,
                    "expert": added_expert_count / added_transition_count,
                },
            },
        },
    }


def _dataset_summary(
    *,
    augmentation_report: AugmentationResult,
    source_episode_count: int,
    source_transition_count: int,
) -> dict:
    added_episode_count = len(augmentation_report.episode_dataset.episode_end_indices)
    added_transition_count = int(augmentation_report.episode_dataset.states.shape[0])

    added_synthetic_count = (
        augmentation_report.counts.accepted * augmentation_report.synthetic_part_length
    )
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
            "by_source": {
                "count": {
                    "synthetic": added_synthetic_count,
                    "expert": dataset_expert_count,
                },
                "fraction": {
                    "synthetic": added_synthetic_count / dataset_transition_count,
                    "expert": dataset_expert_count / dataset_transition_count,
                },
            },
        },
    }
