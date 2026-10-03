from time import perf_counter

import torch
from wandb import Run

from scripts.environments.condition import Condition
from scripts.environments.environment import Environment
from scripts.evaluation.config import EvaluationConfig, resolve_evaluation_stage
from scripts.evaluation.core.rollout import Episode
from scripts.evaluation.core.scoring import (
    RewardStatistics,
    ScoringSession,
    SuccessStatistics,
    calculate_reward_statistics,
    calculate_success_statistics,
)
from scripts.evaluation.rollouts_artifact import upload_evaluation_rollouts
from scripts.shared import stage_names
from scripts.shared.log import create_logger
from scripts.shared.policy_types import PolicyFactory

logger = create_logger()


STAGE_NAME = stage_names.EVALUATION

ALL_CONDITIONS_LABEL = "0_all_conditions"


def evaluate_policy(
    *,
    environment: Environment,
    policy_factory: PolicyFactory,
    config: EvaluationConfig,
    run: Run,
) -> None:
    reward_means: list[float] = []
    episode_records_by_condition_label: dict[str, list[Episode]] = {}
    stage = resolve_evaluation_stage(stage=config.stage)

    with (
        torch.no_grad(),
        ScoringSession(
            environment=environment,
            parallel_environment_count=config.parallel_environment_count,
        ) as scoring_session,
    ):
        scoring_start = perf_counter()
        for (
            condition,
            episode_records,
        ) in scoring_session.roll_out_episodes_with_the_same_seeds_in_every_condition(
            policy_factory=policy_factory,
            start_episode_seed=stage.start_episode_seed,
            episode_count_per_condition=stage.episode_count_per_condition,
        ):
            run.summary[f"{STAGE_NAME}/{condition.label}/7_scoring_seconds"] = (
                perf_counter() - scoring_start
            )
            logger.info("Scored the policy in %s.", condition.label)

            reward = calculate_reward_statistics(episode_records=episode_records)
            reward_means.append(reward.mean)
            episode_records_by_condition_label[condition.label] = episode_records

            _write_summary(
                run=run,
                condition=condition,
                reward=reward,
                success=calculate_success_statistics(episode_records=episode_records),
            )
            scoring_start = perf_counter()

    run.summary[f"{STAGE_NAME}/{ALL_CONDITIONS_LABEL}/1_reward_mean"] = sum(
        reward_means
    ) / len(reward_means)

    if config.save_rollouts:
        upload_evaluation_rollouts(
            episode_records_by_condition_label=episode_records_by_condition_label,
            run=run,
        )


# ---- Private --------------------------------------------


def _write_summary(
    *,
    run: Run,
    condition: Condition,
    reward: RewardStatistics,
    success: SuccessStatistics,
) -> None:
    run.summary[f"{STAGE_NAME}/{condition.label}/1_reward_mean"] = reward.mean
    run.summary[f"{STAGE_NAME}/{condition.label}/2_reward_std"] = reward.std
    run.summary[f"{STAGE_NAME}/{condition.label}/3_reward_standard_error"] = (
        reward.standard_error
    )
    run.summary[f"{STAGE_NAME}/{condition.label}/4_success_rate"] = success.rate
    run.summary[f"{STAGE_NAME}/{condition.label}/5_success_standard_error"] = (
        success.standard_error
    )
