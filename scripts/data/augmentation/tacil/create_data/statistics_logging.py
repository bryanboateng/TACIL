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
from scripts.shared.distribution_summary import summarize_distribution  # noqa: E402

STAGE_NAME = "2_data"

_CLIPPING_PERCENTILE = 99.5


def log_rollout_statistics(
    *,
    rollout_diagnostics: RolloutDiagnostics,
    median_expert_step_distance: float,
    uncertainty_limit: float,
    run: Run,
) -> None:
    termination_reasons = (
        rollout_diagnostics.termination_reasons_batch.cpu().numpy()
    )  # (anchor_count,)

    run.summary[f"{STAGE_NAME}/1_rollout/median_step_distance"] = (
        median_expert_step_distance
    )
    run.summary[f"{STAGE_NAME}/1_rollout/uncertainty_limit"] = uncertainty_limit
    run.summary[f"{STAGE_NAME}/1_rollout/anchor_count"] = int(
        termination_reasons.shape[0]
    )
    for termination_reason in TerminationReason:
        run.summary[f"{STAGE_NAME}/2_termination/{termination_reason.name.lower()}"] = (
            int((termination_reasons == termination_reason).sum())
        )

    step_uncertainties = rollout_diagnostics.step_uncertainties.cpu().numpy()
    if step_uncertainties.size > 0:
        for label, value in summarize_distribution(values=step_uncertainties).items():
            run.summary[f"{STAGE_NAME}/5_step_uncertainties/{label}"] = value
        run.log(
            data={
                f"{STAGE_NAME}/5_step_uncertainties/plot": _distribution_panel(
                    values=step_uncertainties, column_name="step uncertainty"
                )
            }
        )


def log_dataset_statistics(
    *,
    augmentation_result: AugmentationResult,
    source_episode_count: int,
    source_transition_count: int,
    run: Run,
) -> None:
    synthetic_part_lengths = augmentation_result.synthetic_part_lengths
    counts = augmentation_result.counts

    # Every anchor is either accepted or empty, so the two together are the
    # anchor count.
    anchor_count = counts.accepted + counts.empty
    run.summary[f"{STAGE_NAME}/1_rollout/accepted_count"] = counts.accepted
    run.summary[f"{STAGE_NAME}/1_rollout/accepted_fraction"] = (
        counts.accepted / anchor_count
    )

    augmented_transition_count = int(augmentation_result.episode_store.states.shape[0])
    synthetic_transition_count = int(synthetic_part_lengths.sum())
    for label, value in summarize_distribution(values=synthetic_part_lengths).items():
        run.summary[f"{STAGE_NAME}/3_synthetic/length/{label}"] = value
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
            f"{STAGE_NAME}/3_synthetic/length/plot": _distribution_panel(
                values=synthetic_part_lengths, column_name="synthetic part length"
            )
        }
    )


def _bins(*, values: np.ndarray) -> list[float] | int:
    if np.issubdtype(values.dtype, np.integer):
        # One bar per integer value, centered on it.
        return np.arange(values.min() - 0.5, values.max() + 1.5).tolist()

    edges = np.linspace(float(values.min()), float(values.max()), num=21)
    return edges.tolist() if np.all(np.diff(edges) > 0.0) else 1


def _distribution_panel(*, values: np.ndarray, column_name: str) -> wandb.Image:
    clipping_limit = float(np.percentile(a=values, q=_CLIPPING_PERCENTILE))
    clipped_values = values[values <= clipping_limit]  # (clipped_count,)

    figure, axes = plt.subplots(figsize=(6, 4))
    axes.hist(clipped_values, bins=_bins(values=clipped_values), edgecolor="white")
    # The tail is cut off to keep the bars readable, so the axis has to say so
    # — otherwise the picture quietly contradicts the logged maximum.
    axes.set_xlabel(
        xlabel=f"{column_name} (longest {100.0 - _CLIPPING_PERCENTILE:g}% not shown)"
    )
    axes.set_ylabel(ylabel="count")
    figure.tight_layout()

    image = wandb.Image(figure)
    plt.close(fig=figure)
    return image
