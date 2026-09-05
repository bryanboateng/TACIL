from dataclasses import dataclass


@dataclass(frozen=True)
class ObservationNoiseConfig:
    initial_std: float = 0.01
    decay_rate: float = 0.995
    minimum_std: float = 0.001

    def __post_init__(self) -> None:
        for name, value in (
            ("initial_std", self.initial_std),
            ("minimum_std", self.minimum_std),
        ):
            if value < 0:
                raise ValueError(f"{name} must not be negative, but it is {value}.")

        if self.decay_rate <= 0:
            raise ValueError(
                f"decay_rate must be positive, but it is {self.decay_rate}."
            )

        if self.decay_rate > 1.0:
            raise ValueError(
                f"decay_rate must be at most 1, but it is {self.decay_rate}."
            )

        if self.minimum_std > self.initial_std:
            raise ValueError(
                f"minimum_std ({self.minimum_std}) must not exceed initial_std "
                f"({self.initial_std})."
            )


@dataclass(frozen=True)
class Config:
    observation_noise: ObservationNoiseConfig
    plan_length: int = 8
    progression_weight: float = 1.0
    attraction_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.plan_length < 2:
            raise ValueError(
                f"plan_length must be at least 2, but it is {self.plan_length}."
            )

        for name, value in (
            ("progression_weight", self.progression_weight),
            ("attraction_weight", self.attraction_weight),
        ):
            if value < 0:
                raise ValueError(f"{name} must not be negative, but it is {value}.")
