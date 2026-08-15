from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    action_noise_range_fraction: float = 0.1
    uncertainty_limit_step_multiple: float = 0.5
