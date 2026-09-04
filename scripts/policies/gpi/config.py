from dataclasses import dataclass


@dataclass(frozen=True)
class ObservationNoiseConfig:
    initial_std: float = 0.01
    decay_rate: float = 0.995
    minimum_std: float = 0.001

    def __post_init__(self) -> None:
        # Zero noise is a fair choice; a negative spread is not a spread.
        for name, value in (
            ("initial_std", self.initial_std),
            ("minimum_std", self.minimum_std),
        ):
            if value < 0:
                raise ValueError(f"{name} must not be negative, but it is {value}.")
        # The noise decays toward the floor, so a rate above one would let it
        # grow instead, and the floor has to lie under the starting point.
        if self.decay_rate <= 0:
            raise ValueError(
                f"decay_rate must be positive, but it is {self.decay_rate}."
            )
        if self.decay_rate > 1.0:
            raise ValueError(
                f"decay_rate must be at most 1, but it is {self.decay_rate}; "
                f"above that the noise grows instead of decaying."
            )
        if self.minimum_std > self.initial_std:
            raise ValueError(
                f"minimum_std ({self.minimum_std}) must not exceed initial_std "
                f"({self.initial_std}), or the noise starts below its floor."
            )


@dataclass(frozen=True)
class Config:
    observation_noise: ObservationNoiseConfig
    plan_length: int = 8
    progression_weight: float = 1.0
    attraction_weight: float = 1.0

    def __post_init__(self) -> None:
        # A plan of zero steps would never advance through a demonstration.
        # The weights scale a pull each: at zero the policy stands still, below
        # zero it moves away from what it is following.
        for name, value in (
            ("plan_length", self.plan_length),
            ("progression_weight", self.progression_weight),
            ("attraction_weight", self.attraction_weight),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, but it is {value}.")
