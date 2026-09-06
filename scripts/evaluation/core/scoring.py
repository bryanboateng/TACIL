from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch

from scripts.evaluation.core.condition import ALL_CONDITIONS, Condition
from scripts.evaluation.core.rollout import (
    EpisodeRecord,
    create_environment,
    roll_out_episodes,
)
from scripts.shared.policy_types import PolicyFactory


@dataclass(frozen=True)
class RewardStatistics:
    mean: float
    # The spread of the episodes around the mean.
    std: float
    # The uncertainty of the mean itself. A comparison of two methods needs
    # this value, not the spread of the episodes.
    standard_error: float
    episode_count: int


@dataclass(frozen=True)
class SuccessStatistics:
    rate: float
    # The uncertainty of the rate. Successes are counted, not averaged, so the
    # spread follows from the rate itself rather than from the episodes.
    standard_error: float


@torch.no_grad()
def iterate_scored_episodes(
    *,
    policy_factory: PolicyFactory,
    start_episode_seed: int,
    episode_count_per_condition: int,
) -> Iterator[tuple[Condition, list[EpisodeRecord]]]:
    for condition in ALL_CONDITIONS:
        yield (
            condition,
            _run_scored_episodes(
                policy_factory=policy_factory,
                start_episode_seed=start_episode_seed,
                episode_count=episode_count_per_condition,
                condition=condition,
            ),
        )


def _run_scored_episodes(
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
            policy=policy_factory(episode_seeds=seeds),
            render=False,
        )
    finally:
        for environment in environments:
            environment.close()


@torch.no_grad()
def compute_reward_statistics(
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

    # ddof=1, because these episodes are a sample of the possible seeds. They
    # are not the full population.
    std = float(rewards.std(ddof=1)) if episode_count > 1 else 0.0
    return RewardStatistics(
        mean=float(rewards.mean()),
        std=std,
        standard_error=std / float(np.sqrt(episode_count)),
        episode_count=episode_count,
    )


def compute_success_statistics(
    *, episode_records: list[EpisodeRecord]
) -> SuccessStatistics:
    if not episode_records:
        return SuccessStatistics(rate=0.0, standard_error=float("nan"))

    rate = float(np.mean([record.terminated for record in episode_records]))
    # An episode either succeeds or it does not, so the rate is a proportion
    # and its spread follows from the rate: p * (1 - p) per episode.
    return SuccessStatistics(
        rate=rate,
        standard_error=float(np.sqrt(rate * (1.0 - rate) / len(episode_records))),
    )


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
