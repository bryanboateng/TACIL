"""Exercise scoring workers without training a policy or connecting to W&B."""

import cProfile
import os
import platform
import pstats
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import gymnasium as gym
import numpy as np
import tyro

from scripts.environments.registry import EnvironmentName, create_environment
from scripts.evaluation.config import VALIDATION_START_SEED
from scripts.evaluation.core.parallel import ParallelEnvironmentPool
from scripts.evaluation.core.reset_diagnostics import (
    ResetLineProfiler,
    print_reset_line_profiles,
)
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
    # Limit a diagnostic run without changing normal validation.
    condition_count: int | None = None
    # Save worker reset profiles and print the combined timing breakdown.
    profile_directory: Path | None = None
    trace_reset: bool = False

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


class _ProfiledReset(gym.Wrapper):
    def __init__(self, environment: gym.Env, path: Path, trace_reset: bool) -> None:
        super().__init__(environment)
        self._profile = cProfile.Profile()
        self._line_profile = ResetLineProfiler() if trace_reset else None
        self._path = path

    def reset(self, **kwargs):
        try:
            if self._line_profile is not None:
                return self._line_profile.run(self.env.reset, **kwargs)
            return self._profile.runcall(self.env.reset, **kwargs)
        finally:
            # Save after each reset so profiles survive later worker failures.
            if self._line_profile is not None:
                self._line_profile.save(self._path.with_suffix(".json"))
            else:
                self._profile.dump_stats(str(self._path))


def _short_environment(
    *,
    factory: Callable[[], gym.Env],
    step_count: int,
    profile_directory: Path | None,
    condition_index: int,
    trace_reset: bool,
) -> gym.Env:
    environment = gym.wrappers.TimeLimit(factory(), max_episode_steps=step_count)
    if profile_directory is not None:
        environment = _ProfiledReset(
            environment,
            profile_directory
            / f"worker-{os.getpid()}-condition-{condition_index}.prof",
            trace_reset,
        )
    return environment


def _policy_factory(*, action: np.ndarray, episode_seeds: list[int]) -> Policy:
    def policy(
        *, observations_by_episode_index: dict[int, np.ndarray]
    ) -> dict[int, np.ndarray]:
        return {i: action.copy() for i in observations_by_episode_index}

    return policy


def main() -> None:
    config = tyro.cli(Config)
    logger.info(
        "Python %s; packages: %s",
        platform.python_version(),
        {name: version(name) for name in ("sapien", "mani-skill", "torch", "numpy")},
    )
    logger.info(
        "CPU affinity: %s; load averages: %s",
        sorted(os.sched_getaffinity(0)),
        os.getloadavg(),
    )
    for path in ("/proc/meminfo", "/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu.stat"):
        try:
            lines = Path(path).read_text().splitlines()
            if path.endswith("meminfo"):
                lines = [
                    line
                    for line in lines
                    if line.startswith(("MemAvailable:", "SwapFree:", "SwapTotal:"))
                ]
            logger.info("%s: %s", path, "; ".join(lines))
        except OSError:
            pass
    environment = create_environment(name=config.environment)
    conditions = environment.conditions[: config.condition_count]
    profile_directory = None
    requested_directory = config.profile_directory or (
        Path("output/reset-trace") if config.trace_reset else None
    )
    if requested_directory is not None:
        requested_directory.mkdir(parents=True, exist_ok=True)
        profile_directory = Path(
            tempfile.mkdtemp(prefix="reset-", dir=requested_directory)
        )
        logger.info("Saving reset profiles to %s", profile_directory)
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
                profile_directory=profile_directory,
                condition_index=i,
                trace_reset=config.trace_reset,
            )
            for i, condition in enumerate(conditions)
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
    if profile_directory is not None:
        if config.trace_reset:
            print_reset_line_profiles(profile_directory)
        profiles = sorted(profile_directory.glob("*.prof"))
        if profiles:
            logger.info(
                "Reset timings summed across workers (not elapsed wall time). "
                "First table: total time including callees; second: time inside each function."
            )
            stats = pstats.Stats(*(str(path) for path in profiles)).strip_dirs()
            stats.sort_stats("cumulative").print_stats(25)
            stats.sort_stats("tottime").print_stats(25)


if __name__ == "__main__":
    main()
