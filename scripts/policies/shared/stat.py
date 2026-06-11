from dataclasses import dataclass


@dataclass
class Checkpoint:
    validation_max_reward_mean: float
    epoch: int
    state_dict: dict
