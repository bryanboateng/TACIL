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
STAGE_NAME = "2_data"

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

    run.summary[f"{STAGE_NAME}/1_rollout/median_step_distance"] = (
        median_expert_step_distance
    )
    run.summary[f"{STAGE_NAME}/1_rollout/error_limit"] = error_limit
    run.summary[f"{STAGE_NAME}/1_rollout/anchor_count"] = int(
        termination_reasons.shape[0]
    )
    for termination_reason in TerminationReason:
        run.summary[f"{STAGE_NAME}/2_termination/{termination_reason.name.lower()}"] = (
            int((termination_reasons == termination_reason).sum())
        )

    histogram_panels: dict[str, wandb.Image] = {}
    for name, values_batch in (
        (
            "accumulated_error_estimates",
            rollout_diagnostics.accumulated_error_estimates,
        ),
        ("step_error_estimates", rollout_diagnostics.step_error_estimates),
        (
            "uncertainties_about_next_state",
            rollout_diagnostics.uncertainties_about_next_state,
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
            histogram_panels[f"{STAGE_NAME}/5_panels/{name}"] = _distribution_panel(
                values=values, column_name=name
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
    run.summary[f"{STAGE_NAME}/1_rollout/accepted_count"] = counts.accepted
    run.summary[f"{STAGE_NAME}/1_rollout/accepted_fraction"] = counts.accepted / (
        counts.accepted + counts.empty
    )

    augmented_transition_count = int(augmentation_result.episode_store.states.shape[0])
    synthetic_transition_count = int(synthetic_part_lengths.sum())
    run.summary[f"{STAGE_NAME}/3_synthetic/length_mean"] = float(
        synthetic_part_lengths.mean()
    )
    run.summary[f"{STAGE_NAME}/3_synthetic/length_std"] = float(
        synthetic_part_lengths.std()
    )
    run.summary[f"{STAGE_NAME}/3_synthetic/length_max"] = int(
        synthetic_part_lengths.max()
    )
    run.summary[f"{STAGE_NAME}/3_synthetic/transition_count"] = (
        synthetic_transition_count
    )
    run.summary[f"{STAGE_NAME}/3_synthetic/copied_expert_transition_count"] = (
        augmented_transition_count - synthetic_transition_count
    )

    combined_transition_count = source_transition_count + augmented_transition_count
    run.summary[f"{STAGE_NAME}/4_dataset/episode_count"] = (
        source_episode_count + counts.accepted
    )
    run.summary[f"{STAGE_NAME}/4_dataset/transition_count"] = combined_transition_count
    run.summary[f"{STAGE_NAME}/4_dataset/synthetic_fraction"] = (
        synthetic_transition_count / combined_transition_count
    )

    run.log(
        data={
            f"{STAGE_NAME}/5_panels/synthetic_lengths": _distribution_panel(
                values=synthetic_part_lengths, column_name="synthetic_part_length"
            )
        }
    )


def _bins(*, values: np.ndarray) -> np.ndarray | int:
    """Bin edges that are guaranteed to be usable.

    Values that are all the same, or so close together that twenty edges
    between them do not come out distinct in floating point, get a single
    bin — matplotlib would otherwise refuse to draw at all.
    """
    if np.issubdtype(values.dtype, np.integer):
        # One bar per integer value, centered on it.
        return np.arange(values.min() - 0.5, values.max() + 1.5)

    edges = np.linspace(float(values.min()), float(values.max()), num=21)
    return edges if np.all(np.diff(edges) > 0.0) else 1


def _distribution_panel(*, values: np.ndarray, column_name: str) -> wandb.Image:
    """Histogram rendered with matplotlib and logged as an image.

    None of wandb's interactive chart types draws a one-off distribution
    readably (its histogram bins in the browser, its bar chart is horizontal
    with transposed axes), so we render the figure ourselves."""
    # A run whose budget rejects everything, or whose model blows up, can
    # leave these values infinite or all the same. Neither is a reason to
    # lose the trial, so the panel says so instead of raising.
    finite_values = values[np.isfinite(values)]  # (finite_count,)
    dropped_count = int(values.size - finite_values.size)

    figure, axes = plt.subplots(figsize=(6, 4))
    if finite_values.size == 0:
        axes.set_title(label=f"no finite values ({values.size} discarded)")
    else:
        clipped_values = finite_values[
            finite_values <= float(np.percentile(a=finite_values, q=99.5))
        ]  # (clipped_count,)
        axes.hist(clipped_values, bins=_bins(values=clipped_values), edgecolor="white")

    axes.set_xlabel(
        xlabel=column_name
        if dropped_count == 0
        else f"{column_name} ({dropped_count} non-finite discarded)"
    )
    axes.set_ylabel(ylabel="count")
    figure.tight_layout()

    image = wandb.Image(figure)
    plt.close(fig=figure)
    return image
