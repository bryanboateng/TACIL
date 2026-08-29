from wandb import Run

from scripts.data.augmentation.distillation.rollout import RolloutDiagnostics
from scripts.data.augmentation.tacil.create_data.statistics_logging import STAGE_NAME


def log_rollout_statistics(
    *,
    rollout_diagnostics: RolloutDiagnostics,
    median_expert_step_distance: float,
    on_manifold_tolerance: float,
    run: Run,
) -> None:
    run.summary[f"{STAGE_NAME}/1_rollout/median_step_distance"] = (
        median_expert_step_distance
    )
    run.summary[f"{STAGE_NAME}/1_rollout/on_manifold_tolerance"] = on_manifold_tolerance
    run.summary[f"{STAGE_NAME}/1_rollout/chain_count"] = rollout_diagnostics.chain_count
    # Chains whose disturbance did not push them off the demonstrations. A large
    # share means the disturbance is too weak to be worth recovering from.
    run.summary[f"{STAGE_NAME}/1_rollout/already_on_manifold_count"] = (
        rollout_diagnostics.already_on_manifold_count
    )
    run.summary[f"{STAGE_NAME}/1_rollout/returned_count"] = (
        rollout_diagnostics.returned_count
    )
    run.summary[f"{STAGE_NAME}/1_rollout/capped_count"] = (
        rollout_diagnostics.capped_count
    )

    # How often GPI found its way back within the length it was given. It says
    # nothing about whether the data is kept -- every chain is -- but a low
    # share means the chains end far from anything recorded.
    followed_count = (
        rollout_diagnostics.chain_count - rollout_diagnostics.already_on_manifold_count
    )
    if followed_count > 0:
        run.summary[f"{STAGE_NAME}/1_rollout/return_rate"] = (
            rollout_diagnostics.returned_count / followed_count
        )
