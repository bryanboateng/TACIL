from wandb import Run

from scripts.data.augmentation.tacil.create_data.backward_rollout import (
    RolloutDiagnostics,
    TerminationReason,
)
from scripts.data.augmentation.tacil.create_data.episode_assembly import (
    AugmentationResult,
)
from scripts.shared.distribution_panel import distribution_panel
from scripts.shared.distribution_summary import summarize_distribution

STAGE_NAME = "2_data"


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
        run.summary[f"{STAGE_NAME}/3_termination/{termination_reason.name.lower()}"] = (
            int((termination_reasons == termination_reason).sum())
        )

    step_uncertainties = rollout_diagnostics.step_uncertainties.cpu().numpy()
    if step_uncertainties.size > 0:
        for label, value in summarize_distribution(values=step_uncertainties).items():
            run.summary[f"{STAGE_NAME}/2_step_uncertainties/{label}"] = value
        run.log(
            data={
                f"{STAGE_NAME}/2_step_uncertainties/plot": distribution_panel(
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

    # An empty anchor is one whose very first backward step was rejected, so
    # nothing was invented in front of it. Zero is the good outcome, and the
    # denominator is the anchor_count logged next to it.
    anchor_count = counts.accepted + counts.empty
    run.summary[f"{STAGE_NAME}/1_rollout/empty_anchor_count"] = counts.empty
    run.summary[f"{STAGE_NAME}/1_rollout/empty_anchor_fraction"] = (
        counts.empty / anchor_count
    )

    augmented_transition_count = int(augmentation_result.episode_store.states.shape[0])
    synthetic_transition_count = int(synthetic_part_lengths.sum())
    for label, value in summarize_distribution(values=synthetic_part_lengths).items():
        run.summary[f"{STAGE_NAME}/4_synthetic/length/{label}"] = value
    run.summary[f"{STAGE_NAME}/4_synthetic/transition_count"] = (
        synthetic_transition_count
    )
    run.summary[f"{STAGE_NAME}/4_synthetic/copied_expert_transition_count"] = (
        augmented_transition_count - synthetic_transition_count
    )

    combined_transition_count = source_transition_count + augmented_transition_count
    run.summary[f"{STAGE_NAME}/5_dataset/episode_count"] = (
        source_episode_count + counts.accepted
    )
    run.summary[f"{STAGE_NAME}/5_dataset/transition_count"] = combined_transition_count
    run.summary[f"{STAGE_NAME}/5_dataset/synthetic_transition_fraction"] = (
        synthetic_transition_count / combined_transition_count
    )

    run.log(
        data={
            f"{STAGE_NAME}/4_synthetic/length/plot": distribution_panel(
                values=synthetic_part_lengths, column_name="synthetic part length"
            )
        }
    )
