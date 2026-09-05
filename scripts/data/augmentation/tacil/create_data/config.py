from dataclasses import dataclass
from typing import Literal, TypeAlias

ChainErrorCriterion: TypeAlias = Literal["account", "replay"]


@dataclass(frozen=True)
class DataConfig:
    chain_error: ChainErrorCriterion
    action_noise_range_fraction: float
    chain_error_limit_step_multiple: float

    def __post_init__(self) -> None:
        if self.action_noise_range_fraction < 0:
            raise ValueError(
                f"action_noise_range_fraction must not be negative, but it is "
                f"{self.action_noise_range_fraction}."
            )

        if self.chain_error_limit_step_multiple <= 0:
            raise ValueError(
                f"chain_error_limit_step_multiple must be positive, but it is "
                f"{self.chain_error_limit_step_multiple}."
            )
