from dataclasses import dataclass
from typing import Literal, TypeAlias

StoppingRule: TypeAlias = Literal["account", "replay"]


@dataclass(frozen=True)
class DataConfig:
    action_noise: float
    stopping_rule: StoppingRule
    error_limit_in_steps: float

    def __post_init__(self) -> None:
        if self.action_noise < 0:
            raise ValueError(
                f"action_noise must not be negative, but it is {self.action_noise}."
            )

        if self.error_limit_in_steps <= 0:
            raise ValueError(
                f"error_limit_in_steps must be positive, but it is "
                f"{self.error_limit_in_steps}."
            )
