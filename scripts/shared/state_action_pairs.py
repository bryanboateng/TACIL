from dataclasses import dataclass

import numpy as np
from numpy import typing as npt


@dataclass
class StateActionPairs:
    states: npt.NDArray[np.float32]  # (pair_count, state_dimensionality)
    actions: npt.NDArray[np.float32]  # (pair_count, action_dimensionality)
