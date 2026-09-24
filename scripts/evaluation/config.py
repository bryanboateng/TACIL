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
    # Maximum unique episodes to replay for each condition's video.
    video_episode_count: int = 3
    # CPU simulation processes; 1 uses the sequential rollout path.
    parallel_environment_count: int = 4

    def __post_init__(self) -> None:
        for name, value in (
            ("video_episode_count", self.video_episode_count),
            ("parallel_environment_count", self.parallel_environment_count),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, but it is {value}.")
