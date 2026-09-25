import os
from dataclasses import dataclass, field
from typing import Annotated, TypeAlias

import tyro

VALIDATION_START_SEED = 10_000
TUNING_START_SEED = 20_000
FINAL_START_SEED = 30_000


def calculate_default_parallel_environment_count() -> int:
    # Each worker simulates on one thread, and the parent process runs the policy,
    # so it keeps a core to itself. Logical CPUs are halved because two
    # hyperthreads on one core simulate slower than one thread does.
    return max(1, (os.process_cpu_count() or 2) // 2 - 1)


@dataclass(frozen=True)
class TuningEvaluation:
    pass


@dataclass(frozen=True)
class FinalEvaluation:
    pass


@dataclass(frozen=True)
class CustomEvaluation:
    start_episode_seed: int
    episode_count_per_condition: int

    def __post_init__(self) -> None:
        if self.start_episode_seed < 0:
            raise ValueError(
                "start_episode_seed must not be negative, but it is "
                f"{self.start_episode_seed}."
            )
        if self.episode_count_per_condition <= 0:
            raise ValueError(
                "episode_count_per_condition must be positive, but it is "
                f"{self.episode_count_per_condition}."
            )


EvaluationStage: TypeAlias = (
    Annotated[TuningEvaluation, tyro.conf.subcommand(name="tuning")]
    | Annotated[FinalEvaluation, tyro.conf.subcommand(name="final")]
    | Annotated[CustomEvaluation, tyro.conf.subcommand(name="custom")]
)


@dataclass(frozen=True)
class EvaluationConfig:
    stage: EvaluationStage
    record_videos: bool
    # CPU simulation processes; 1 uses the sequential rollout path.
    parallel_environment_count: int = field(
        default_factory=calculate_default_parallel_environment_count
    )

    def __post_init__(self) -> None:
        if self.parallel_environment_count <= 0:
            raise ValueError(
                "parallel_environment_count must be positive, but it is "
                f"{self.parallel_environment_count}."
            )
