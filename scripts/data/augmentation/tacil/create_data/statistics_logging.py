import numpy as np
import wandb
from wandb import Run

from scripts.data.augmentation.tacil.create_data.backward_rollout import (
    BackwardTrajectories,
    RolloutDiagnostics,
    TerminationReason,
)
from scripts.data.augmentation.tacil.create_data.episode_assembly import (
    AugmentationResult,
)


def log_rollout_statistics(
    *,
    backward_trajectories: BackwardTrajectories,
    rollout_diagnostics: RolloutDiagnostics,
    median_expert_step_distance: float,
    error_limit: float,
    run: Run,
) -> None:
    step_count_batch = backward_trajectories.step_count_batch.cpu()  # (anchor_count,)
    termination_reason_batch = (
        rollout_diagnostics.termination_reasons_batch.cpu()
    )  # (anchor_count,)
    anchor_count = int(step_count_batch.shape[0])

    run.summary["median_expert_step_distance"] = median_expert_step_distance
    run.summary["error_limit"] = error_limit
    run.summary["rollout"] = {
        "anchor_count": anchor_count,
        "step_count": {
            "mean": float(step_count_batch.float().mean().item()),
            "max": int(step_count_batch.max().item()),
        },
        "anchors_without_steps": int((step_count_batch == 0).sum().item()),
        "termination_reasons": {
            termination_reason.name.lower(): int(
                (termination_reason_batch == termination_reason).sum().item()
            )
            for termination_reason in TerminationReason
        },
    }

    figures = {
        "figures/step_counts": wandb.Histogram(step_count_batch.numpy().tolist())
    }
    for figure_name, values_batch in (
        (
            "accumulated_error_estimates",
            rollout_diagnostics.accumulated_error_estimates,
        ),
        ("step_error_estimates", rollout_diagnostics.step_error_estimates),
        (
            "disagreements_about_next_state",
            rollout_diagnostics.disagreements_about_next_state,
        ),
        (
            "sensitivity_scaled_step_distances",
            rollout_diagnostics.sensitivity_scaled_step_distances,
        ),
        (
            "local_transition_lipschitz_constants",
            rollout_diagnostics.local_transition_lipschitz_constants,
        ),
    ):
        values = values_batch.cpu().numpy()
        if values.size > 0:
            figures[f"figures/{figure_name}"] = _clipped_histogram(values=values)

    run.log(data=figures, step=0)


def log_dataset_statistics(
    *,
    augmentation_result: AugmentationResult,
    source_episode_count: int,
    source_transition_count: int,
    run: Run,
) -> None:
    synthetic_part_lengths = augmentation_result.synthetic_part_lengths
    counts = augmentation_result.counts
    anchor_count = counts.accepted + counts.empty

    added_episode_count = len(augmentation_result.episode_store.episode_end_indices)
    added_transition_count = int(augmentation_result.episode_store.states.shape[0])
    added_synthetic_count = int(synthetic_part_lengths.sum())
    added_expert_count = added_transition_count - added_synthetic_count
    dataset_transition_count = source_transition_count + added_transition_count

    run.summary["anchor"] = {
        "total_count": anchor_count,
        "accepted_count": counts.accepted,
        "empty_count": counts.empty,
        "accepted_fraction": counts.accepted / anchor_count,
    }
    run.summary["synthetic_part_length"] = {
        "mean": float(synthetic_part_lengths.mean()),
        "max": int(synthetic_part_lengths.max()),
    }
    run.summary["added"] = {
        "episode_count": added_episode_count,
        "transition_count": added_transition_count,
        "synthetic_transition_count": added_synthetic_count,
        "expert_transition_count": added_expert_count,
    }
    run.summary["dataset"] = {
        "episode_count": source_episode_count + added_episode_count,
        "transition_count": dataset_transition_count,
        "synthetic_fraction": added_synthetic_count / dataset_transition_count,
    }

    run.log(
        data={
            "figures/synthetic_part_lengths": wandb.Histogram(
                synthetic_part_lengths.tolist()
            )
        },
        step=0,
    )


def _clipped_histogram(*, values: np.ndarray) -> wandb.Histogram:  # (value_count,)
    clipped_values = values[values <= float(np.percentile(a=values, q=99.5))]

    # "auto" can pick more bins than wandb allows, so cap the bin count.
    bin_count = min(len(np.histogram_bin_edges(clipped_values, bins="auto")) - 1, 512)
    return wandb.Histogram(np_histogram=np.histogram(a=clipped_values, bins=bin_count))
