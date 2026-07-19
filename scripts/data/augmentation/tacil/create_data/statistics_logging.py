import numpy as np
import wandb
from wandb import Run
from wandb.plot.custom_chart import CustomChart

from scripts.data.augmentation.tacil.create_data.backward_rollout import (
    RolloutDiagnostics,
    TerminationReason,
)
from scripts.data.augmentation.tacil.create_data.episode_assembly import (
    AugmentationResult,
)

_MAX_HISTOGRAM_VALUE_COUNT = 10_000


def log_rollout_statistics(
    *,
    rollout_diagnostics: RolloutDiagnostics,
    median_expert_step_distance: float,
    error_limit: float,
    run: Run,
) -> None:
    termination_reasons = (
        rollout_diagnostics.termination_reasons_batch.cpu().numpy()
    )  # (anchor_count,)

    run.summary["median_expert_step_distance"] = median_expert_step_distance
    run.summary["error_limit"] = error_limit
    run.summary["anchor_count"] = int(termination_reasons.shape[0])
    for termination_reason in TerminationReason:
        run.summary[f"termination_reason/{termination_reason.name.lower()}"] = int(
            (termination_reasons == termination_reason).sum()
        )

    histogram_panels: dict[str, CustomChart] = {}
    for name, values_batch in (
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
            histogram_panels[f"histograms/{name}"] = _histogram_panel(
                values=values, column_name=name
            )

    run.log(data=histogram_panels, step=0)


def log_dataset_statistics(
    *,
    augmentation_result: AugmentationResult,
    source_episode_count: int,
    source_transition_count: int,
    run: Run,
) -> None:
    synthetic_part_lengths = augmentation_result.synthetic_part_lengths
    counts = augmentation_result.counts

    run.summary["anchors/accepted_count"] = counts.accepted
    run.summary["anchors/empty_count"] = counts.empty
    run.summary["anchors/accepted_fraction"] = counts.accepted / (
        counts.accepted + counts.empty
    )

    run.summary["synthetic_part_length/mean"] = float(synthetic_part_lengths.mean())
    run.summary["synthetic_part_length/max"] = int(synthetic_part_lengths.max())

    augmented_episode_count = len(augmentation_result.episode_store.episode_end_indices)
    augmented_transition_count = int(augmentation_result.episode_store.states.shape[0])
    synthetic_transition_count = int(synthetic_part_lengths.sum())
    run.summary["augmented_episodes/count"] = augmented_episode_count
    run.summary["augmented_episodes/transition_count"] = augmented_transition_count
    run.summary["augmented_episodes/synthetic_transition_count"] = (
        synthetic_transition_count
    )
    run.summary["augmented_episodes/copied_expert_transition_count"] = (
        augmented_transition_count - synthetic_transition_count
    )

    combined_transition_count = source_transition_count + augmented_transition_count
    run.summary["combined_dataset/episode_count"] = (
        source_episode_count + augmented_episode_count
    )
    run.summary["combined_dataset/transition_count"] = combined_transition_count
    run.summary["combined_dataset/synthetic_transition_fraction"] = (
        synthetic_transition_count / combined_transition_count
    )

    run.log(
        data={
            "histograms/synthetic_part_lengths": _histogram_panel(
                values=synthetic_part_lengths, column_name="synthetic_part_length"
            )
        },
        step=0,
    )


def _histogram_panel(*, values: np.ndarray, column_name: str) -> CustomChart:
    """A static histogram panel built from a table of raw values.

    wandb's native Histogram type is meant to be logged once per training step
    and renders as a step-indexed heatmap; for a one-off distribution that
    display is unreadable, so we use a custom chart instead."""
    clipped_values = values[
        values <= float(np.percentile(a=values, q=99.5))
    ]  # (clipped_count,)
    if clipped_values.size > _MAX_HISTOGRAM_VALUE_COUNT:
        clipped_values = np.random.default_rng(seed=0).choice(
            a=clipped_values, size=_MAX_HISTOGRAM_VALUE_COUNT, replace=False
        )

    return wandb.plot.histogram(
        table=wandb.Table(
            data=[[float(value)] for value in clipped_values], columns=[column_name]
        ),
        value=column_name,
        title=column_name,
    )
