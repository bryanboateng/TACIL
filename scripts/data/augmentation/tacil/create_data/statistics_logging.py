from wandb import Run

from scripts.data.augmentation.tacil.create_data.backward_rollout import (
    RolloutDiagnostics,
)
from scripts.data.augmentation.tacil.create_data.chain_error import (
    ChainErrorCriterion,
    describe_criterion,
)
from scripts.data.augmentation.tacil.create_data.episode_assembly import (
    AugmentationResult,
)
from scripts.shared.create_distribution_panel import create_distribution_panel
from scripts.shared.distribution_summary import summarize_distribution

STAGE_NAME = "2_data"


def log_rollout_statistics(
    *,
    rollout_diagnostics: RolloutDiagnostics,
    median_expert_step_distance: float,
    chain_error_criterion: ChainErrorCriterion,
    chain_error_limit: float,
    run: Run,
) -> None:
    run.summary[f"{STAGE_NAME}/1_rollout/median_step_distance"] = (
        median_expert_step_distance
    )
    run.summary[f"{STAGE_NAME}/1_rollout/chain_error_criterion"] = describe_criterion(
        criterion=chain_error_criterion
    )
    run.summary[f"{STAGE_NAME}/1_rollout/chain_error_limit"] = chain_error_limit
    run.summary[f"{STAGE_NAME}/1_rollout/anchor_count"] = (
        rollout_diagnostics.anchor_count
    )
    run.summary[f"{STAGE_NAME}/1_rollout/not_converged_count"] = (
        rollout_diagnostics.not_converged_count
    )

    # Each chain's last accepted value, as a share of the limit. Values close
    # to 1 mean the limit is what ended the walk; a distribution sitting far
    # below 1 means the chains stopped for some other reason.
    limit_shares = (
        rollout_diagnostics.final_chain_errors.cpu().numpy() / chain_error_limit
    )
    if limit_shares.size > 0:
        for label, value in summarize_distribution(values=limit_shares).items():
            run.summary[f"{STAGE_NAME}/2_limit_tightness/{label}"] = value
        run.log(
            data={
                f"{STAGE_NAME}/2_limit_tightness/plot": create_distribution_panel(
                    values=limit_shares, column_name="final chain error / limit"
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

    for label, value in summarize_distribution(values=synthetic_part_lengths).items():
        run.summary[f"{STAGE_NAME}/4_synthetic_length/{label}"] = value

    # The three kinds of transition the policy trains on, counted so that they
    # add up to the total and nothing has to be worked out by subtraction.
    #
    # Expert steps are counted twice on purpose, because the policy really does
    # see them twice: once in the recorded episodes, and once as the run-out
    # copied behind each anchor so that a synthetic prefix leads back into
    # recorded behaviour.
    invented_count = int(synthetic_part_lengths.sum())
    run_out_count = (
        int(augmentation_result.episode_store.states.shape[0]) - invented_count
    )
    total_count = source_transition_count + invented_count + run_out_count

    run.summary[f"{STAGE_NAME}/5_dataset/1_recorded_expert_transitions"] = (
        source_transition_count
    )
    run.summary[f"{STAGE_NAME}/5_dataset/2_invented_transitions"] = invented_count
    run.summary[f"{STAGE_NAME}/5_dataset/3_copied_expert_run_out_transitions"] = (
        run_out_count
    )
    run.summary[f"{STAGE_NAME}/5_dataset/4_total_transitions"] = total_count
    run.summary[f"{STAGE_NAME}/5_dataset/5_invented_fraction"] = (
        invented_count / total_count
    )
    run.summary[f"{STAGE_NAME}/5_dataset/6_episode_count"] = (
        source_episode_count + counts.accepted
    )

    run.log(
        data={
            f"{STAGE_NAME}/4_synthetic_length/plot": create_distribution_panel(
                values=synthetic_part_lengths, column_name="synthetic part length"
            )
        }
    )
