from wandb import Run

from scripts.data.augmentation.tacil.create_data.backward_rollout import (
    RolloutDiagnostics,
)
from scripts.data.augmentation.tacil.create_data.episode_assembly import (
    AugmentationResult,
)
from scripts.data.augmentation.tacil.create_data.nearest_expert_state import (
    NearestExpertStateDistances,
)
from scripts.data.augmentation.tacil.create_data.nearest_expert_state_plot import (
    create_nearest_expert_state_plot,
)
from scripts.shared import stage_names
from scripts.shared.distribution_plot import create_distribution_plot
from scripts.shared.distribution_summary import summarize_distribution

STAGE_NAME = stage_names.DATA


def log_median_expert_step_distances(
    *,
    median_expert_state_step_distance: float,
    median_expert_action_step_distance: float,
    run: Run,
) -> None:
    run.summary[f"{STAGE_NAME}/1_median_expert_step_distance/1_state"] = (
        median_expert_state_step_distance
    )
    run.summary[f"{STAGE_NAME}/1_median_expert_step_distance/2_action"] = (
        median_expert_action_step_distance
    )


def log_rollout_statistics(
    *, rollout_diagnostics: RolloutDiagnostics, error_limit: float, run: Run
) -> None:
    run.summary[f"{STAGE_NAME}/2_rollout/1_error_limit"] = error_limit
    run.summary[f"{STAGE_NAME}/2_rollout/2_anchor_count"] = (
        rollout_diagnostics.anchor_count
    )
    run.summary[f"{STAGE_NAME}/2_rollout/3_not_converged_count"] = (
        rollout_diagnostics.not_converged_count
    )


def log_dataset_statistics(
    *,
    augmentation_result: AugmentationResult,
    source_episode_count: int,
    source_transition_count: int,
    synthetic_distances: NearestExpertStateDistances,
    expert_distances: NearestExpertStateDistances,
    median_expert_state_step_distance: float,
    median_expert_action_step_distance: float,
    run: Run,
) -> None:
    trajectory_lengths = augmentation_result.trajectory_lengths
    counts = augmentation_result.counts

    anchor_count = counts.accepted + counts.empty
    run.summary[f"{STAGE_NAME}/2_rollout/4_empty_anchor_count"] = counts.empty
    run.summary[f"{STAGE_NAME}/2_rollout/5_empty_anchor_fraction"] = (
        counts.empty / anchor_count
    )

    for label, value in summarize_distribution(values=trajectory_lengths).items():
        run.summary[f"{STAGE_NAME}/3_synthetic_length/{label}"] = value

    synthetic_count = int(trajectory_lengths.sum())
    run_out_states_count = (
        int(augmentation_result.episode_store.states.shape[0]) - synthetic_count
    )
    total_count = source_transition_count + synthetic_count + run_out_states_count

    run.summary[f"{STAGE_NAME}/4_dataset/1_expert_transition_count"] = (
        source_transition_count
    )
    run.summary[f"{STAGE_NAME}/4_dataset/2_synthetic_transition_count"] = (
        synthetic_count
    )
    run.summary[f"{STAGE_NAME}/4_dataset/3_copied_expert_run_out_transition_count"] = (
        run_out_states_count
    )
    run.summary[f"{STAGE_NAME}/4_dataset/4_total_transition_count"] = total_count
    run.summary[f"{STAGE_NAME}/4_dataset/5_total_episode_count"] = (
        source_episode_count + counts.accepted
    )
    run.summary[f"{STAGE_NAME}/4_dataset/6_synthetic_to_expert_transition_ratio"] = (
        synthetic_count / source_transition_count
    )

    synthetic_state_distances = (
        synthetic_distances.state_distances.cpu().numpy()
        / median_expert_state_step_distance
    )
    expert_state_distances = (
        expert_distances.state_distances.cpu().numpy()
        / median_expert_state_step_distance
    )
    synthetic_action_distances = (
        synthetic_distances.action_distances.cpu().numpy()
        / median_expert_action_step_distance
    )
    expert_action_distances = (
        expert_distances.action_distances.cpu().numpy()
        / median_expert_action_step_distance
    )

    for quantity, values_per_group in (
        (
            "1_state_distance",
            (
                ("1_synthetic", synthetic_state_distances),
                ("2_expert", expert_state_distances),
            ),
        ),
        (
            "2_action_distance",
            (
                ("1_synthetic", synthetic_action_distances),
                ("2_expert", expert_action_distances),
            ),
        ),
    ):
        for group, values in values_per_group:
            for label, value in summarize_distribution(values=values).items():
                run.summary[
                    f"{STAGE_NAME}/5_nearest_expert_state/{quantity}/{group}/{label}"
                ] = value

    run.log(
        data={
            f"{STAGE_NAME}/3_synthetic_length/4_plot": create_distribution_plot(
                values=trajectory_lengths, xlabel="synthetic part length"
            ),
            f"{STAGE_NAME}/5_nearest_expert_state/3_plot": (
                create_nearest_expert_state_plot(
                    synthetic_state_distances=synthetic_state_distances,
                    synthetic_action_distances=synthetic_action_distances,
                    expert_state_distances=expert_state_distances,
                    expert_action_distances=expert_action_distances,
                )
            ),
        }
    )
