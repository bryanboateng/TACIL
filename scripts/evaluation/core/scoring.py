from dataclasses import dataclass

import numpy as np
import torch

from scripts.evaluation.core.condition import Condition
from scripts.evaluation.core.rollout import (
    EpisodeRecord,
    create_environment,
    roll_out_episodes,
)
from scripts.shared.policy_types import PolicyFactory


@dataclass(frozen=True)
class RewardStatistics:
    mean: float
    std: float


@torch.no_grad()
def run_scored_episodes(
    *,
    policy_factory: PolicyFactory,
    start_episode_seed: int,
    episode_count: int,
    condition: Condition,
) -> list[EpisodeRecord]:
    seeds = list(range(start_episode_seed, start_episode_seed + episode_count))
    environments = [
        create_environment(condition=condition, render_settings=None) for _ in seeds
    ]
    try:
        return roll_out_episodes(
            environments=environments,
            seeds=seeds,
            policy=policy_factory(seeds=seeds),
            render=False,
        )
    finally:
        for environment in environments:
            environment.close()


def reward_statistics(episode_records: list[EpisodeRecord]) -> RewardStatistics:
    rewards = np.array([record.reward for record in episode_records])
    return RewardStatistics(mean=float(rewards.mean()), std=float(rewards.std()))


def success_rate(episode_records: list[EpisodeRecord]) -> float:
    if not episode_records:
        return 0.0
    return float(np.mean([record.terminated for record in episode_records]))


def pick_representative_episodes(
    *, episode_records: list[EpisodeRecord]
) -> dict[int, EpisodeRecord]:
    count = 9

    sorted_records = sorted(episode_records, key=lambda record: record.reward)
    last_index = len(sorted_records) - 1
    representative_records: dict[int, EpisodeRecord] = {}
    for step in range(count):
        fraction = step / (count - 1)
        percentile = round(number=100 * fraction)
        representative_records[percentile] = sorted_records[
            round(number=last_index * fraction)
        ]
    return representative_records
