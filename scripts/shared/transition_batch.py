from dataclasses import dataclass

import numpy as np
from numpy import typing as npt


@dataclass
class TransitionBatch:
    states: npt.NDArray[np.float32]  # (transition_count, state_dimensionality)
    actions: npt.NDArray[np.float32]  # (transition_count, action_dimensionality)
