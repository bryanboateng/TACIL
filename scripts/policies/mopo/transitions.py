from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Transitions:
    """Steps a policy can learn from, recorded or imagined.

    Held as device tensors. The recorded ones are read from a file and moved
    over once; the imagined ones are produced on the device and never leave
    it.
    """

    states: torch.Tensor  # (transition_count, state_dimensionality)
    actions: torch.Tensor  # (transition_count, action_dimensionality)
    next_states: torch.Tensor  # (transition_count, state_dimensionality)
    rewards: torch.Tensor  # (transition_count, 1)
    terminations: torch.Tensor  # (transition_count, 1)
