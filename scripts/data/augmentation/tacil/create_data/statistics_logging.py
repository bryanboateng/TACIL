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

# Summary groups carry numeric prefixes so wandb's alphabetical summary table
# shows them in pipeline order: error model -> rollout -> termination ->
# augmented episodes -> combined dataset.


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

    run.summary["2_rollout/median_expert_step_distance"] = median_expert_step_distance
    run.summary["2_rollout/error_limit"] = error_limit
    run.summary["2_rollout/anchor_count"] = int(termination_reasons.shape[0])
    for termination_reason in TerminationReason:
        run.summary[f"3_termination_reason/{termination_reason.name.lower()}"] = int(
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
            histogram_panels[f"distributions/{name}"] = _distribution_panel(
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

    # counts.accepted equals the number of augmented episodes, and
    # counts.empty equals anchor_count - accepted: each value appears once.
    run.summary["2_rollout/accepted_anchor_count"] = counts.accepted
    run.summary["2_rollout/accepted_anchor_fraction"] = counts.accepted / (
        counts.accepted + counts.empty
    )

    augmented_transition_count = int(augmentation_result.episode_store.states.shape[0])
    synthetic_transition_count = int(synthetic_part_lengths.sum())
    run.summary["4_augmented_episodes/synthetic_part_length_mean"] = float(
        synthetic_part_lengths.mean()
    )
    run.summary["4_augmented_episodes/synthetic_part_length_max"] = int(
        synthetic_part_lengths.max()
    )
    run.summary["4_augmented_episodes/synthetic_transition_count"] = (
        synthetic_transition_count
    )
    run.summary["4_augmented_episodes/copied_expert_transition_count"] = (
        augmented_transition_count - synthetic_transition_count
    )

    combined_transition_count = source_transition_count + augmented_transition_count
    run.summary["5_combined_dataset/episode_count"] = (
        source_episode_count + counts.accepted
    )
    run.summary["5_combined_dataset/transition_count"] = combined_transition_count
    run.summary["5_combined_dataset/synthetic_transition_fraction"] = (
        synthetic_transition_count / combined_transition_count
    )

    run.log(
        data={
            "distributions/synthetic_part_lengths": _distribution_panel(
                values=synthetic_part_lengths, column_name="synthetic_part_length"
            )
        },
        step=0,
    )


def _distribution_panel(*, values: np.ndarray, column_name: str) -> CustomChart:
    """Distribution panel with the value on the x axis and the count on the y
    axis. Binned here in numpy — wandb's built-in histogram types bin in the
    browser and render one-off distributions unreadably, and its bar preset
    only draws horizontal bars with the axes transposed."""
    clipped_values = values[
        values <= float(np.percentile(a=values, q=99.5))
    ]  # (clipped_count,)

    if np.issubdtype(clipped_values.dtype, np.integer):
        # One point per occurring integer value.
        bin_values, bin_counts = np.unique(clipped_values, return_counts=True)
        rows = [
            [int(bin_value), int(bin_count)]
            for bin_value, bin_count in zip(bin_values, bin_counts)
        ]
    else:
        bin_counts, bin_edges = np.histogram(a=clipped_values, bins=20)
        bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2  # (20,)
        rows = [
            [float(f"{bin_midpoint:.4g}"), int(bin_count)]
            for bin_midpoint, bin_count in zip(bin_midpoints, bin_counts)
        ]

    return wandb.plot.line(
        table=wandb.Table(data=rows, columns=[column_name, "count"]),
        x=column_name,
        y="count",
        title=column_name,
    )
