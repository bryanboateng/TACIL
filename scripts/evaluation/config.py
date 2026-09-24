from dataclasses import dataclass
from enum import Enum

VALIDATION_START_SEED = 10_000
TUNING_START_SEED = 20_000
FINAL_START_SEED = 30_000


class EvaluationStage(Enum):
    TUNING = "tuning"
    FINAL = "final"


@dataclass(frozen=True)
class EvaluationConfig:
    stage: EvaluationStage
    # CPU simulation processes; 1 uses the sequential rollout path.
    parallel_environment_count: int = 4

    def __post_init__(self) -> None:
        if self.parallel_environment_count <= 0:
            raise ValueError(
                "parallel_environment_count must be positive, but it is "
                f"{self.parallel_environment_count}."
            )
