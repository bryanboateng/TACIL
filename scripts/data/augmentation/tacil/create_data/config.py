from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    action_noise_fraction: float = 0.1
    error_limit_fraction: float = 0.5
