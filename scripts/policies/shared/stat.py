from dataclasses import dataclass


@dataclass
class Checkpoint:
    validation_reward_mean: float
    validation_reward_std: float
    epoch: int
    state_dict: dict
