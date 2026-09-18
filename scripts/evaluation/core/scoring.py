from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from scripts.environments.condition import Condition
from scripts.environments.environment import Environment
from scripts.evaluation.core.rollout import EpisodeRecord, roll_out_episodes
from scripts.shared.policy_types import PolicyFactory


@dataclass(frozen=True)
class RewardStatistics:
    mean: float
    std: float
    standard_error: float
    episode_count: int


@dataclass(frozen=True)
class SuccessStatistics:
    rate: float
    standard_error: float


def run_scored_episodes_per_condition(
    *,
    environment: Environment,
    policy_factory: PolicyFactory,
    start_episode_seed: int,
    episode_count_per_condition: int,
) -> Iterator[tuple[Condition, list[EpisodeRecord]]]:
    for condition in environment.conditions:
        yield (
            condition,
            _run_scored_episodes(
                environment=environment,
                policy_factory=policy_factory,
                start_episode_seed=start_episode_seed,
                episode_count=episode_count_per_condition,
                condition=condition,
            ),
        )


def run_scored_episodes_over_conditions(
    *,
    environment: Environment,
    policy_factory: PolicyFactory,
    episode_count_per_condition: int,
) -> list[EpisodeRecord]:
    return [
        episode_record
        for condition_index, condition in enumerate(iterable=environment.conditions)
        for episode_record in _run_scored_episodes(
            environment=environment,
            policy_factory=policy_factory,
            start_episode_seed=(10_000 + condition_index * episode_count_per_condition),
            episode_count=episode_count_per_condition,
            condition=condition,
        )
    ]


def _run_scored_episodes(
    *,
    environment: Environment,
    policy_factory: PolicyFactory,
    start_episode_seed: int,
    episode_count: int,
    condition: Condition,
) -> list[EpisodeRecord]:
    seeds = list(range(start_episode_seed, start_episode_seed + episode_count))
    gym_environments = [
        environment.create_gym_environment(condition=condition, render_settings=None)
        for _ in seeds
    ]
    try:
        return roll_out_episodes(
            environments=gym_environments,
            seeds=seeds,
            policy=policy_factory(episode_seeds=seeds),
            render=False,
        )
    finally:
        for gym_environment in gym_environments:
            gym_environment.close()


def calculate_reward_statistics(
    *, episode_records: list[EpisodeRecord]
) -> RewardStatistics:
    rewards = np.array([record.reward for record in episode_records])
    episode_count = int(rewards.shape[0])
    if episode_count == 0:
        return RewardStatistics(
            mean=float("nan"),
            std=float("nan"),
            standard_error=float("nan"),
            episode_count=0,
        )

    std = float(rewards.std(ddof=1)) if episode_count > 1 else 0.0
    return RewardStatistics(
        mean=float(rewards.mean()),
        std=std,
        standard_error=std / float(np.sqrt(episode_count)),
        episode_count=episode_count,
    )


def calculate_success_statistics(
    *, episode_records: list[EpisodeRecord]
) -> SuccessStatistics:
    episode_count = len(episode_records)
    if episode_count == 0:
        return SuccessStatistics(rate=float("nan"), standard_error=float("nan"))

    rate = float(np.mean([record.terminated for record in episode_records]))
    return SuccessStatistics(
        rate=rate, standard_error=float(np.sqrt(rate * (1.0 - rate) / episode_count))
    )


def pick_representative_episodes(
    *, episode_records: list[EpisodeRecord]
) -> dict[int, EpisodeRecord]:
    percentiles = (0, 2, 5, 10, 20, 40, 50, 60, 80, 90, 95, 98, 100)

    sorted_records = sorted(episode_records, key=lambda record: record.reward)
    last_index = len(sorted_records) - 1
    return {
        percentile: sorted_records[round(number=last_index * percentile / 100)]
        for percentile in percentiles
    }
