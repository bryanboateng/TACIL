from dataclasses import dataclass
from typing import Literal

ErrorSignal = Literal["uncertainty", "sensitivity_scaled_step_distance"]


@dataclass(frozen=True)
class DataConfig:
    action_noise_fraction: float = 0.1
    error_limit_fraction: float = 0.5

    error_signal: ErrorSignal = "uncertainty"
