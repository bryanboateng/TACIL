from dataclasses import dataclass
from typing import Literal

from wandb import Run

from scripts.shared import stage_names


@dataclass
class Checkpoint:
    validation_reward_mean: float
    validation_reward_std: float
    validation_success_rate: float
    epoch: int
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


def summarize_best_checkpoint(*, run: Run, best_checkpoint: Checkpoint) -> None:
    section = f"{stage_names.POLICY}/3_best_checkpoint"
    run.summary[f"{section}/1_epoch"] = best_checkpoint.epoch
    run.summary[f"{section}/2_validation_reward_mean"] = (
        best_checkpoint.validation_reward_mean
    )
    run.summary[f"{section}/3_validation_reward_std"] = (
        best_checkpoint.validation_reward_std
    )
    run.summary[f"{section}/4_validation_success_rate"] = (
        best_checkpoint.validation_success_rate
    )
