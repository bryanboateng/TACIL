from collections.abc import Generator
from dataclasses import dataclass
from functools import partial
from typing import Self

import numpy as np

from scripts.environments.condition import Condition
from scripts.environments.environment import Environment
from scripts.evaluation.core.parallel import ParallelEnvironmentPool
from scripts.evaluation.core.rollout import Episode
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


class ScoringSession:
    def __init__(
        self, *, environment: Environment, parallel_environment_count: int
    ) -> None:

        if parallel_environment_count < 1:
            raise ValueError("parallel_environment_count must be positive")

        self.environment = environment
        self._parallel_environment_count = parallel_environment_count
        self._parallel_environment_pool: ParallelEnvironmentPool | None = None
        self._is_closed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self._is_closed = True
        if self._parallel_environment_pool is not None:
            self._parallel_environment_pool.close()
            self._parallel_environment_pool = None

    def roll_out_episodes_with_separate_seeds_per_condition(
        self,
        *,
        policy_factory: PolicyFactory,
        start_episode_seed: int,
        episode_count_per_condition: int,
    ) -> list[Episode]:
        return [
            record
            for condition_index in range(len(self.environment.conditions))
            for record in self.roll_out_episodes_in_one_condition(
                condition_index=condition_index,
                policy_factory=policy_factory,
                start_episode_seed=(
                    start_episode_seed + condition_index * episode_count_per_condition
                ),
                episode_count=episode_count_per_condition,
            )
        ]

    def roll_out_episodes_with_the_same_seeds_in_every_condition(
        self,
        *,
        policy_factory: PolicyFactory,
        start_episode_seed: int,
        episode_count_per_condition: int,
    ) -> Generator[tuple[Condition, list[Episode]], None, None]:
        for condition_index, condition in enumerate(self.environment.conditions):
            yield (
                condition,
                self.roll_out_episodes_in_one_condition(
                    condition_index=condition_index,
                    policy_factory=policy_factory,
                    start_episode_seed=start_episode_seed,
                    episode_count=episode_count_per_condition,
                ),
            )

    def roll_out_episodes_in_one_condition(
        self,
        *,
        condition_index: int,
        policy_factory: PolicyFactory,
        start_episode_seed: int,
        episode_count: int,
    ) -> list[Episode]:
        if self._is_closed:
            raise RuntimeError("Scoring session is closed")
        if self._parallel_environment_pool is None:
            self._parallel_environment_pool = ParallelEnvironmentPool(
                factories=[
                    partial(
                        self.environment.create_gym_environment,
                        condition=condition,
                        render_settings=None,
                    )
                    for condition in self.environment.conditions
                ],
                worker_count=self._parallel_environment_count,
            )
        return self._parallel_environment_pool.roll_out(
            seeds=list(range(start_episode_seed, start_episode_seed + episode_count)),
            policy_factory=policy_factory,
            condition_index=condition_index,
        )


def roll_out_episodes_with_the_same_seeds_in_every_condition(
    *,
    environment: Environment,
    policy_factory: PolicyFactory,
    start_episode_seed: int,
    episode_count_per_condition: int,
    parallel_environment_count: int,
) -> Generator[tuple[Condition, list[Episode]], None, None]:
    with ScoringSession(
        environment=environment, parallel_environment_count=parallel_environment_count
    ) as scoring_session:
        yield from scoring_session.roll_out_episodes_with_the_same_seeds_in_every_condition(
            policy_factory=policy_factory,
            start_episode_seed=start_episode_seed,
            episode_count_per_condition=episode_count_per_condition,
        )


def calculate_reward_statistics(*, episode_records: list[Episode]) -> RewardStatistics:
    rewards = np.array([record.reward for record in episode_records])
    episode_count = int(rewards.shape[0])
    if episode_count == 0:
        return RewardStatistics(
            mean=float("nan"),
            std=float("nan"),
            standard_error=float("nan"),
            episode_count=0,
        )
    else:
        std = float(rewards.std(ddof=1)) if episode_count > 1 else 0.0
        return RewardStatistics(
            mean=float(rewards.mean()),
            std=std,
            standard_error=std / float(np.sqrt(episode_count)),
            episode_count=episode_count,
        )


def calculate_success_statistics(
    *, episode_records: list[Episode]
) -> SuccessStatistics:
    episode_count = len(episode_records)
    if episode_count == 0:
        return SuccessStatistics(rate=float("nan"), standard_error=float("nan"))
    else:
        rate = float(np.mean([record.terminated for record in episode_records]))
        return SuccessStatistics(
            rate=rate,
            standard_error=float(np.sqrt(rate * (1.0 - rate) / episode_count)),
        )


def pick_representative_episodes(
    *, episode_records: list[Episode]
) -> dict[int, Episode]:
    percentiles = (0, 2, 5, 10, 20, 40, 50, 60, 80, 90, 95, 98, 100)

    if not episode_records:
        return {}

    sorted_records = sorted(episode_records, key=lambda record: record.reward)
    last_index = len(sorted_records) - 1
    return {
        percentile: sorted_records[round(number=last_index * percentile / 100)]
        for percentile in percentiles
    }
