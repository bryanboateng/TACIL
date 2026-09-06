from wandb import Run

from scripts.data.augmentation.tacil.create_data.backward_rollout import (
    RolloutDiagnostics,
)
from scripts.data.augmentation.tacil.create_data.episode_assembly import (
    AugmentationResult,
)
from scripts.data.augmentation.tacil.create_data.nearest_recorded_state import (
    NearestRecordedStateDistances,
)
from scripts.data.augmentation.tacil.create_data.nearest_recorded_state_plot import (
    create_nearest_recorded_state_plot,
)
from scripts.shared import stage_names
from scripts.shared.distribution_plot import create_distribution_plot
from scripts.shared.distribution_summary import summarize_distribution

STAGE_NAME = stage_names.DATA


def log_rollout_statistics(
    *,
    rollout_diagnostics: RolloutDiagnostics,
    expert_step_distance: float,
    error_limit: float,
    run: Run,
) -> None:
    run.summary[f"{STAGE_NAME}/1_rollout/1_median_step_distance"] = expert_step_distance
    run.summary[f"{STAGE_NAME}/1_rollout/2_error_limit"] = error_limit
    run.summary[f"{STAGE_NAME}/1_rollout/3_anchor_count"] = (
        rollout_diagnostics.anchor_count
    )
    run.summary[f"{STAGE_NAME}/1_rollout/4_not_converged_count"] = (
        rollout_diagnostics.not_converged_count
    )

    # Each trajectory's last accepted value, as a share of the limit. Values close
    # to 1 mean the limit is what ended the walk; a distribution sitting far
    # below 1 means the trajectories stopped for some other reason.
    limit_shares = rollout_diagnostics.final_errors.cpu().numpy() / error_limit
    if limit_shares.size > 0:
        for label, value in summarize_distribution(values=limit_shares).items():
            run.summary[f"{STAGE_NAME}/2_limit_tightness/{label}"] = value
        run.log(
            data={
                f"{STAGE_NAME}/2_limit_tightness/4_plot": create_distribution_plot(
                    values=limit_shares, column_name="final trajectory error / limit"
                )
            }
        )


def log_dataset_statistics(
    *,
    augmentation_result: AugmentationResult,
    source_episode_count: int,
    source_transition_count: int,
    synthetic_distances: NearestRecordedStateDistances,
    recorded_distances: NearestRecordedStateDistances,
    expert_step_distance: float,
    run: Run,
) -> None:
    trajectory_lengths = augmentation_result.trajectory_lengths
    counts = augmentation_result.counts

    # An empty anchor is one whose very first backward step was rejected, so
    # nothing was invented in front of it. Zero is the good outcome, and the
    # denominator is the anchor_count logged next to it.
    anchor_count = counts.accepted + counts.empty
    run.summary[f"{STAGE_NAME}/1_rollout/5_empty_anchor_count"] = counts.empty
    run.summary[f"{STAGE_NAME}/1_rollout/6_empty_anchor_fraction"] = (
        counts.empty / anchor_count
    )

    for label, value in summarize_distribution(values=trajectory_lengths).items():
        run.summary[f"{STAGE_NAME}/4_synthetic_length/{label}"] = value

    # The three kinds of transition the policy trains on, counted so that they
    # add up to the total and nothing has to be worked out by subtraction.
    #
    # Expert steps are counted twice on purpose, because the policy really does
    # see them twice: once in the recorded episodes, and once as the run-out
    # copied behind each anchor so that a synthetic prefix leads back into
    # recorded behaviour.
    invented_count = int(trajectory_lengths.sum())
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

    synthetic_state_distances = (
        synthetic_distances.state_distances.cpu().numpy() / expert_step_distance
    )
    recorded_state_distances = (
        recorded_distances.state_distances.cpu().numpy() / expert_step_distance
    )
    synthetic_action_distances = synthetic_distances.action_distances.cpu().numpy()
    recorded_action_distances = recorded_distances.action_distances.cpu().numpy()

    for quantity, values_per_group in (
        (
            "1_state_distance",
            (
                ("1_synthetic", synthetic_state_distances),
                ("2_recorded", recorded_state_distances),
            ),
        ),
        (
            "2_action_distance",
            (
                ("1_synthetic", synthetic_action_distances),
                ("2_recorded", recorded_action_distances),
            ),
        ),
    ):
        for group, values in values_per_group:
            for label, value in summarize_distribution(values=values).items():
                run.summary[
                    f"{STAGE_NAME}/6_nearest_recorded_state/{quantity}/{group}/{label}"
                ] = value

    run.log(
        data={
            f"{STAGE_NAME}/4_synthetic_length/4_plot": create_distribution_plot(
                values=trajectory_lengths, column_name="synthetic part length"
            ),
            f"{STAGE_NAME}/6_nearest_recorded_state/3_plot": (
                create_nearest_recorded_state_plot(
                    synthetic_state_distances=synthetic_state_distances,
                    synthetic_action_distances=synthetic_action_distances,
                    recorded_state_distances=recorded_state_distances,
                    recorded_action_distances=recorded_action_distances,
                )
            ),
        }
    )
