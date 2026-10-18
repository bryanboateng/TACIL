from dataclasses import dataclass
from typing import Literal, TypeAlias

StoppingRule: TypeAlias = Literal["account", "replay", "ccil"]


@dataclass(frozen=True)
class DataConfig:
    action_noise: float
    stopping_rule: StoppingRule
    error_limit_in_steps: float
    depth_limit: int | None = None

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

        if self.depth_limit is not None and self.depth_limit < 1:
            raise ValueError(
                f"depth_limit must be at least 1 when it is set, but it is "
                f"{self.depth_limit}."
            )
