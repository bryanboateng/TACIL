from collections.abc import Generator
from dataclasses import dataclass
from functools import partial
from typing import Self

import numpy as np

from scripts.environments.condition import Condition
from scripts.environments.environment import Environment
from scripts.evaluation.config import VALIDATION_START_SEED
from scripts.evaluation.core.parallel import ParallelEnvironmentPool
from scripts.evaluation.core.rollout import Episode, roll_out_episodes
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

    def run_over_conditions(
        self, *, policy_factory: PolicyFactory, episode_count_per_condition: int
    ) -> list[Episode]:
        return [
            record
            for condition_index in range(len(self.environment.conditions))
            for record in self.run_episodes_in_condition(
                condition_index=condition_index,
                policy_factory=policy_factory,
                start_episode_seed=(
                    VALIDATION_START_SEED
                    + condition_index * episode_count_per_condition
                ),
                episode_count=episode_count_per_condition,
            )
        ]

    def run_episodes_in_condition(
        self,
        *,
        condition_index: int,
        policy_factory: PolicyFactory,
        start_episode_seed: int,
        episode_count: int,
    ) -> list[Episode]:
        if self._is_closed:
            raise RuntimeError("Scoring session is closed")
        condition = self.environment.conditions[condition_index]
        seeds = list(range(start_episode_seed, start_episode_seed + episode_count))
        if self._parallel_environment_count > 1:
            if self._parallel_environment_pool is None:
                self._parallel_environment_pool = ParallelEnvironmentPool(
                    factories=[
                        partial(
                            self.environment.create_gym_environment,
                            condition=item,
                            render_settings=None,
                        )
                        for item in self.environment.conditions
                    ],
                    worker_count=self._parallel_environment_count,
                )
            return self._parallel_environment_pool.roll_out(
                seeds=seeds,
                policy_factory=policy_factory,
                condition_index=condition_index,
            )
        else:
            return _run_episodes_sequentially(
                environment=self.environment,
                condition=condition,
                seeds=seeds,
                policy_factory=policy_factory,
            )


def run_scored_episodes_per_condition(
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
        for condition_index, condition in enumerate(environment.conditions):
            yield (
                condition,
                scoring_session.run_episodes_in_condition(
                    condition_index=condition_index,
                    policy_factory=policy_factory,
                    start_episode_seed=start_episode_seed,
                    episode_count=episode_count_per_condition,
                ),
            )


def run_scored_episodes_over_conditions(
    *,
    environment: Environment,
    policy_factory: PolicyFactory,
    episode_count_per_condition: int,
    parallel_environment_count: int = 1,
) -> list[Episode]:
    with ScoringSession(
        environment=environment, parallel_environment_count=parallel_environment_count
    ) as session:
        return session.run_over_conditions(
            policy_factory=policy_factory,
            episode_count_per_condition=episode_count_per_condition,
        )


def _run_episodes_sequentially(
    *,
    environment: Environment,
    policy_factory: PolicyFactory,
    seeds: list[int],
    condition: Condition,
) -> list[Episode]:
    gym_environments = []
    try:
        for _ in seeds:
            gym_environments.append(
                environment.create_gym_environment(
                    condition=condition, render_settings=None
                )
            )
        return roll_out_episodes(
            environments=gym_environments,
            seeds=seeds,
            policy=policy_factory(episode_seeds=seeds),
            render=False,
        )
    finally:
        for gym_environment in gym_environments:
            gym_environment.close()


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

    rate = float(np.mean([record.terminated for record in episode_records]))
    return SuccessStatistics(
        rate=rate, standard_error=float(np.sqrt(rate * (1.0 - rate) / episode_count))
    )


def pick_representative_episodes(
    *, episode_records: list[Episode]
) -> dict[int, Episode]:
    if not episode_records:
        return {}
    percentiles = (0, 2, 5, 10, 20, 40, 50, 60, 80, 90, 95, 98, 100)
    sorted_records = sorted(episode_records, key=lambda record: record.reward)
    last_index = len(sorted_records) - 1
    selected: dict[int, Episode] = {}
    selected_indices: set[int] = set()
    for percentile in percentiles:
        index = round(number=last_index * percentile / 100)
        if index not in selected_indices:
            selected[percentile] = sorted_records[index]
            selected_indices.add(index)
    return selected
