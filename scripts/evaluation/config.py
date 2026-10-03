from dataclasses import dataclass
from typing import Annotated, TypeAlias

import tyro

VALIDATION_START_SEED = 10_000
TUNING_START_SEED = 20_000
FINAL_START_SEED = 30_000

TUNING_EPISODE_COUNT_PER_CONDITION = 100
FINAL_EPISODE_COUNT_PER_CONDITION = 200


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
    parallel_environment_count: int = 6
    save_rollouts: bool = False

    def __post_init__(self) -> None:
        if self.parallel_environment_count <= 0:
            raise ValueError(
                "parallel_environment_count must be positive, but it is "
                f"{self.parallel_environment_count}."
            )


@dataclass(frozen=True)
class ResolvedEvaluationStage:
    name: str
    start_episode_seed: int
    episode_count_per_condition: int


def resolve_evaluation_stage(*, stage: EvaluationStage) -> ResolvedEvaluationStage:
    match stage:
        case TuningEvaluation():
            return ResolvedEvaluationStage(
                name="tuning",
                start_episode_seed=TUNING_START_SEED,
                episode_count_per_condition=TUNING_EPISODE_COUNT_PER_CONDITION,
            )
        case FinalEvaluation():
            return ResolvedEvaluationStage(
                name="final",
                start_episode_seed=FINAL_START_SEED,
                episode_count_per_condition=FINAL_EPISODE_COUNT_PER_CONDITION,
            )
        case CustomEvaluation() as custom_evaluation:
            return ResolvedEvaluationStage(
                name="custom",
                start_episode_seed=custom_evaluation.start_episode_seed,
                episode_count_per_condition=(
                    custom_evaluation.episode_count_per_condition
                ),
            )
