from wandb import Run

from scripts.data.augmentation.recovery.planning import PlanningDiagnostics
from scripts.data.augmentation.tacil.create_data.statistics_logging import STAGE_NAME
from scripts.shared.distribution_panel import distribution_panel
from scripts.shared.distribution_summary import summarize_distribution


def log_planning_statistics(
    *,
    planning_diagnostics: PlanningDiagnostics,
    median_expert_step_distance: float,
    replay_error_limit: float,
    run: Run,
) -> None:
    run.summary[f"{STAGE_NAME}/1_rollout/median_step_distance"] = (
        median_expert_step_distance
    )
    run.summary[f"{STAGE_NAME}/1_rollout/replay_error_limit"] = replay_error_limit
    run.summary[f"{STAGE_NAME}/1_rollout/anchor_count"] = (
        planning_diagnostics.anchor_count
    )
    run.summary[f"{STAGE_NAME}/1_rollout/arrived_count"] = (
        planning_diagnostics.arrived_count
    )
    run.summary[f"{STAGE_NAME}/1_rollout/gave_up_count"] = (
        planning_diagnostics.gave_up_count
    )
    run.summary[f"{STAGE_NAME}/1_rollout/replay_rejected_count"] = (
        planning_diagnostics.replay_rejected_count
    )

    # Each accepted chain's replay error, as a share of the limit. A
    # distribution near 1 means the limit is what decides which chains
    # survive; far below 1 means nearly everything the planner finishes also
    # passes the jury.
    limit_shares = (
        planning_diagnostics.final_replay_errors.cpu().numpy() / replay_error_limit
    )
    if limit_shares.size > 0:
        for label, value in summarize_distribution(values=limit_shares).items():
            run.summary[f"{STAGE_NAME}/2_limit_tightness/{label}"] = value
        run.log(
            data={
                f"{STAGE_NAME}/2_limit_tightness/plot": distribution_panel(
                    values=limit_shares, column_name="final replay error / limit"
                )
            }
        )
