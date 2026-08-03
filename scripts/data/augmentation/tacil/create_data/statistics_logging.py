import matplotlib
import numpy as np
import wandb
from wandb import Run

matplotlib.use(backend="Agg")
from matplotlib import pyplot as plt  # noqa: E402

plt.style.use(style="seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300

from scripts.data.augmentation.tacil.create_data.backward_rollout import (  # noqa: E402
    RolloutDiagnostics,
    TerminationReason,
)
from scripts.data.augmentation.tacil.create_data.episode_assembly import (  # noqa: E402
    AugmentationResult,
)

# Every key this stage logs starts with it, so a stage run and a pipeline run
# report the same numbers under the same names. wandb groups panels by the part
# before the first slash, so the stage joins the group with a hyphen rather than
# a slash of its own.
STAGE_PREFIX = "2_data-"

# The groups under it carry numeric prefixes of their own, so wandb's
# alphabetical summary table shows them in pipeline order: error model ->
# rollout -> termination -> augmented episodes -> combined dataset.


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

    run.summary[f"{STAGE_PREFIX}2_rollout/median_expert_step_distance"] = (
        median_expert_step_distance
    )
    run.summary[f"{STAGE_PREFIX}2_rollout/error_limit"] = error_limit
    run.summary[f"{STAGE_PREFIX}2_rollout/anchor_count"] = int(
        termination_reasons.shape[0]
    )
    for termination_reason in TerminationReason:
        run.summary[
            f"{STAGE_PREFIX}3_termination_reason/{termination_reason.name.lower()}"
        ] = int((termination_reasons == termination_reason).sum())

    histogram_panels: dict[str, wandb.Image] = {}
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
            histogram_panels[f"{STAGE_PREFIX}distributions/{name}"] = (
                _distribution_panel(values=values, column_name=name)
            )

    run.log(data=histogram_panels)


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
    run.summary[f"{STAGE_PREFIX}2_rollout/accepted_anchor_count"] = counts.accepted
    run.summary[f"{STAGE_PREFIX}2_rollout/accepted_anchor_fraction"] = (
        counts.accepted / (counts.accepted + counts.empty)
    )

    augmented_transition_count = int(augmentation_result.episode_store.states.shape[0])
    synthetic_transition_count = int(synthetic_part_lengths.sum())
    run.summary[f"{STAGE_PREFIX}4_augmented_episodes/synthetic_part_length_mean"] = (
        float(synthetic_part_lengths.mean())
    )
    run.summary[f"{STAGE_PREFIX}4_augmented_episodes/synthetic_part_length_max"] = int(
        synthetic_part_lengths.max()
    )
    run.summary[f"{STAGE_PREFIX}4_augmented_episodes/synthetic_transition_count"] = (
        synthetic_transition_count
    )
    run.summary[
        f"{STAGE_PREFIX}4_augmented_episodes/copied_expert_transition_count"
    ] = augmented_transition_count - synthetic_transition_count

    combined_transition_count = source_transition_count + augmented_transition_count
    run.summary[f"{STAGE_PREFIX}5_combined_dataset/episode_count"] = (
        source_episode_count + counts.accepted
    )
    run.summary[f"{STAGE_PREFIX}5_combined_dataset/transition_count"] = (
        combined_transition_count
    )
    run.summary[f"{STAGE_PREFIX}5_combined_dataset/synthetic_transition_fraction"] = (
        synthetic_transition_count / combined_transition_count
    )

    run.log(
        data={
            f"{STAGE_PREFIX}distributions/synthetic_part_lengths": _distribution_panel(
                values=synthetic_part_lengths, column_name="synthetic_part_length"
            )
        }
    )


def _distribution_panel(*, values: np.ndarray, column_name: str) -> wandb.Image:
    """Histogram rendered with matplotlib and logged as an image.

    None of wandb's interactive chart types draws a one-off distribution
    readably (its histogram bins in the browser, its bar chart is horizontal
    with transposed axes), so we render the figure ourselves."""
    clipped_values = values[
        values <= float(np.percentile(a=values, q=99.5))
    ]  # (clipped_count,)

    if np.issubdtype(clipped_values.dtype, np.integer):
        # One bar per integer value, centered on it.
        bins = np.arange(clipped_values.min() - 0.5, clipped_values.max() + 1.5)
    else:
        bins = 20

    figure, axes = plt.subplots(figsize=(6, 4))
    axes.hist(clipped_values, bins=bins, edgecolor="white")
    axes.set_xlabel(xlabel=column_name)
    axes.set_ylabel(ylabel="count")
    figure.tight_layout()

    image = wandb.Image(figure)
    plt.close(fig=figure)
    return image
