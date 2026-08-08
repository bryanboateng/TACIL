from wandb import Run

from scripts.evaluation.core.condition import Condition
from scripts.evaluation.core.scoring import (
    RewardStatistics,
    pick_representative_episodes,
    reward_statistics,
    run_scored_episodes,
    success_rate,
)
from scripts.evaluation.core.video import record_condition
from scripts.shared.log import create_logger
from scripts.shared.policy_types import PolicyFactory

logger = create_logger()

# How many episodes a policy is scored over. Enough that the reward mean is
# steady from one evaluation to the next.
DEFAULT_EPISODE_COUNT = 100

# Every key this stage logs starts with it, so a stage run and a pipeline run
# report the same numbers under the same names. wandb groups panels by the part
# before the first slash, so the stage joins the group with a hyphen rather than
# a slash of its own. The sweep optimizes a key built from it, in create_sweep.
STAGE_NAME = "4_evaluation"


def evaluate_policy(
    *,
    policy_factory: PolicyFactory,
    policy_label: str,
    condition: Condition,
    episode_count: int,
    run: Run,
) -> None:
    episode_records = run_scored_episodes(
        policy_factory=policy_factory,
        start_episode_seed=1_000_000,
        episode_count=episode_count,
        condition=condition,
    )

    _write_summary(
        run=run,
        condition=condition,
        reward=reward_statistics(episode_records),
        success_fraction=success_rate(episode_records),
    )
    record_condition(
        policy_label=policy_label,
        condition=condition,
        episode_records=pick_representative_episodes(episode_records=episode_records),
        stage_name=STAGE_NAME,
        run=run,
    )


def _write_summary(
    *, run: Run, condition: Condition, reward: RewardStatistics, success_fraction: float
) -> None:
    run.summary[f"{STAGE_NAME}/{condition.label}/reward_mean"] = reward.mean
    run.summary[f"{STAGE_NAME}/{condition.label}/reward_std"] = reward.std
    run.summary[f"{STAGE_NAME}/{condition.label}/reward_standard_error"] = (
        reward.standard_error
    )
    run.summary[f"{STAGE_NAME}/{condition.label}/episode_count"] = reward.episode_count
    run.summary[f"{STAGE_NAME}/{condition.label}/success_rate"] = success_fraction
