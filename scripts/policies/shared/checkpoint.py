from dataclasses import dataclass
from typing import Literal


@dataclass
class Checkpoint:
    validation_reward_mean: float
    validation_reward_std: float
    validation_success_rate: float
    epoch: int
    step: int
    state_dict: dict


def calculate_checkpoint_selection_key(
    *,
    selection_metric: Literal["reward", "success_then_reward"],
    validation_reward_mean: float,
    validation_success_rate: float,
) -> tuple[float, ...]:
    match selection_metric:
        case "reward":
            return (validation_reward_mean,)
        case "success_then_reward":
            return validation_success_rate, validation_reward_mean
