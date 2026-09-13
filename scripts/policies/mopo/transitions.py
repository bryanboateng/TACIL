from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Transitions:
    states: torch.Tensor  # (transition_count, state_dimensionality)
    actions: torch.Tensor  # (transition_count, action_dimensionality)
    next_states: torch.Tensor  # (transition_count, state_dimensionality)
    rewards: torch.Tensor  # (transition_count, 1)
    terminations: torch.Tensor  # (transition_count, 1)
