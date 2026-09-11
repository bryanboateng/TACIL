from dataclasses import dataclass

import numpy as np
from numpy import typing as npt


@dataclass(frozen=True)
class Transitions:
    states: npt.NDArray[np.float32]  # (transition_count, state_dimensionality)
    actions: npt.NDArray[np.float32]  # (transition_count, action_dimensionality)
    next_states: npt.NDArray[np.float32]  # (transition_count, state_dimensionality)
    rewards: npt.NDArray[np.float32]  # (transition_count, 1)
    terminations: npt.NDArray[np.float32]  # (transition_count, 1)
