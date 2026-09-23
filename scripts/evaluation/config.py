from dataclasses import dataclass

VALIDATION_START_SEED = 10_000


@dataclass(frozen=True)
class EvaluationConfig:
    start_episode_seed: int = 100
    episode_count_per_condition: int = 200
    # Maximum unique episodes to replay for each condition's video.
    video_episode_count: int = 3
    # CPU simulation processes; 1 uses the sequential rollout path.
    parallel_environment_count: int = 4

    def __post_init__(self) -> None:
        for name, value in (
            ("start_episode_seed", self.start_episode_seed),
            ("episode_count_per_condition", self.episode_count_per_condition),
            ("video_episode_count", self.video_episode_count),
            ("parallel_environment_count", self.parallel_environment_count),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, but it is {value}.")
