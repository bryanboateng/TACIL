"""Exercise scoring workers without training a policy or connecting to W&B."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from time import perf_counter

import gymnasium as gym
import numpy as np
import tyro

from scripts.environments.registry import EnvironmentName, create_environment
from scripts.evaluation.config import VALIDATION_START_SEED
from scripts.evaluation.core.parallel import ParallelEnvironmentPool
from scripts.shared.log import create_logger
from scripts.shared.policy_types import Policy

logger = create_logger()


@dataclass(frozen=True)
class Config:
    environment: EnvironmentName = EnvironmentName.PEG_INSERTION
    parallel_environment_count: int = 4
    episode_count_per_condition: int = 50
    step_count: int = 3
    repetition_count: int = 2
    condition_count: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "parallel_environment_count",
            "episode_count_per_condition",
            "step_count",
            "repetition_count",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.condition_count is not None and self.condition_count < 1:
            raise ValueError("condition_count must be positive")


def _short_environment(*, factory: Callable[[], gym.Env], step_count: int) -> gym.Env:
    return gym.wrappers.TimeLimit(factory(), max_episode_steps=step_count)


def _policy_factory(*, action: np.ndarray, episode_seeds: list[int]) -> Policy:
    def policy(
        *, observations_by_episode_index: dict[int, np.ndarray]
    ) -> dict[int, np.ndarray]:
        return {i: action.copy() for i in observations_by_episode_index}

    return policy


def main() -> None:
    config = tyro.cli(Config)
    environment = create_environment(name=config.environment)
    conditions = environment.conditions[: config.condition_count]
    action = (
        np.zeros(7, dtype=np.float32)
        if config.environment == EnvironmentName.PEG_INSERTION
        else np.full(2, 256, dtype=np.float32)
    )
    logger.info(
        "Checking %s: %d workers, %d episodes per condition, %d steps, %d repetitions",
        config.environment,
        config.parallel_environment_count,
        config.episode_count_per_condition,
        config.step_count,
        config.repetition_count,
    )
    reference = None
    with ParallelEnvironmentPool(
        factories=[
            partial(
                _short_environment,
                factory=partial(
                    environment.create_gym_environment,
                    condition=condition,
                    render_settings=None,
                ),
                step_count=config.step_count,
            )
            for condition in conditions
        ],
        worker_count=min(
            config.parallel_environment_count, config.episode_count_per_condition
        ),
    ) as pool:
        for repetition in range(config.repetition_count):
            start = perf_counter()
            records = []
            for i, condition in enumerate(conditions):
                logger.info("Checking condition %s", condition.label)
                first_seed = (
                    VALIDATION_START_SEED + i * config.episode_count_per_condition
                )
                records.extend(
                    pool.roll_out(
                        seeds=list(
                            range(
                                first_seed,
                                first_seed + config.episode_count_per_condition,
                            )
                        ),
                        policy_factory=partial(_policy_factory, action=action),
                        condition_index=i,
                    )
                )
            logger.info(
                "Repetition %d completed in %.2fs",
                repetition + 1,
                perf_counter() - start,
            )
            if reference is None:
                reference = records
            else:
                for original, repeated in zip(reference, records, strict=True):
                    np.testing.assert_array_equal(
                        original.final_observation, repeated.final_observation
                    )
                    np.testing.assert_array_equal(original.rewards, repeated.rewards)
                logger.info("Repeated observations and rewards match exactly")
    logger.info("Worker check passed")


if __name__ == "__main__":
    main()
