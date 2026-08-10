from dataclasses import dataclass


@dataclass(frozen=True)
class ObservationNoiseConfig:
    initial_std: float = 0.01
    decay_rate: float = 0.995
    minimum_std: float = 0.001


@dataclass(frozen=True)
class Config:
    observation_noise: ObservationNoiseConfig
    plan_length: int = 8
    progression_weight: float = 1.0
    attraction_weight: float = 1.0
