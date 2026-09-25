from dataclasses import dataclass
from typing import Literal

from wandb import Run

from scripts.environments.environment import Environment
from scripts.shared import stage_names


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


def summarize_best_checkpoint(
    *, run: Run, best_checkpoint: Checkpoint, environment: Environment
) -> None:
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
    run.summary[f"{section}/5_training_step"] = best_checkpoint.step
    run.summary[f"{section}/6_completed_epoch_count"] = best_checkpoint.epoch + 1
    run.summary[f"{section}/7_selection_metric"] = (
        environment.checkpoint_selection_metric
    )
    run.summary[f"{section}/8_validation_conditions"] = [
        condition.label for condition in environment.conditions
    ]
