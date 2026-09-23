from collections.abc import Generator
from dataclasses import dataclass
from functools import partial
from typing import Self

import numpy as np

from scripts.environments.condition import Condition
from scripts.environments.environment import Environment
from scripts.evaluation.config import VALIDATION_START_SEED
from scripts.evaluation.core.parallel import ParallelEnvironmentPool
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


class ScoringSession:
    """Own one lazy CPU pool for a complete evaluation or training run.

    Workers cache one environment per condition, but reset every episode with
    its original seed. Policies stay in the parent and are rebuilt per batch,
    so validation always uses the current model and fresh policy state.
    """

    def __init__(
        self, *, environment: Environment, parallel_environment_count: int = 1
    ) -> None:
        if parallel_environment_count < 1:
            raise ValueError("parallel_environment_count must be positive")
        self.environment = environment
        self._parallel_environment_count = parallel_environment_count
        self._pool: ParallelEnvironmentPool | None = None
        self._closed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self._closed = True
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    def run_over_conditions(
        self, *, policy_factory: PolicyFactory, episode_count_per_condition: int
    ) -> list[EpisodeRecord]:
        return [
            record
            for condition_index in range(len(self.environment.conditions))
            for record in self.run_condition(
                condition_index=condition_index,
                policy_factory=policy_factory,
                start_episode_seed=(
                    VALIDATION_START_SEED
                    + condition_index * episode_count_per_condition
                ),
                episode_count=episode_count_per_condition,
            )
        ]

    def run_condition(
        self,
        *,
        condition_index: int,
        policy_factory: PolicyFactory,
        start_episode_seed: int,
        episode_count: int,
    ) -> list[EpisodeRecord]:
        if self._closed:
            raise RuntimeError("Scoring session is closed")
        condition = self.environment.conditions[condition_index]
        seeds = list(range(start_episode_seed, start_episode_seed + episode_count))
        if self._parallel_environment_count > 1 and seeds:
            if self._pool is None:
                self._pool = ParallelEnvironmentPool(
                    factories=[
                        partial(
                            self.environment.create_gym_environment,
                            condition=item,
                            render_settings=None,
                        )
                        for item in self.environment.conditions
                    ],
                    worker_count=min(self._parallel_environment_count, len(seeds)),
                )
            return self._pool.roll_out(
                seeds=seeds,
                policy_factory=policy_factory,
                condition_index=condition_index,
            )
        return _run_sequential_episodes(
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
    parallel_environment_count: int = 1,
) -> Generator[tuple[Condition, list[EpisodeRecord]], None, None]:
    with ScoringSession(
        environment=environment, parallel_environment_count=parallel_environment_count
    ) as session:
        for condition_index, condition in enumerate(environment.conditions):
            yield (
                condition,
                session.run_condition(
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
) -> list[EpisodeRecord]:
    with ScoringSession(
        environment=environment, parallel_environment_count=parallel_environment_count
    ) as session:
        return session.run_over_conditions(
            policy_factory=policy_factory,
            episode_count_per_condition=episode_count_per_condition,
        )


def _run_sequential_episodes(
    *,
    environment: Environment,
    policy_factory: PolicyFactory,
    seeds: list[int],
    condition: Condition,
) -> list[EpisodeRecord]:
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
    *, episode_records: list[EpisodeRecord], maximum_episode_count: int | None = None
) -> dict[int, EpisodeRecord]:
    if not episode_records:
        return {}
    if maximum_episode_count is not None and maximum_episode_count <= 0:
        raise ValueError("maximum_episode_count must be positive")
    percentiles = (0, 2, 5, 10, 20, 40, 50, 60, 80, 90, 95, 98, 100)
    if maximum_episode_count is not None:
        count = min(maximum_episode_count, len(episode_records))
        percentiles = (
            (50,)
            if count == 1
            else tuple(np.linspace(0, 100, count).round().astype(int))
        )

    sorted_records = sorted(episode_records, key=lambda record: record.reward)
    last_index = len(sorted_records) - 1
    selected: dict[int, EpisodeRecord] = {}
    selected_indices: set[int] = set()
    for percentile in percentiles:
        index = round(number=last_index * percentile / 100)
        if index not in selected_indices:
            selected[percentile] = sorted_records[index]
            selected_indices.add(index)
    return selected
