from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationConfig:
    start_episode_seed: int
    episode_count_per_condition: int

    def __post_init__(self) -> None:
        for name, value in (
            ("start_episode_seed", self.start_episode_seed),
            ("episode_count_per_condition", self.episode_count_per_condition),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, but it is {value}.")
