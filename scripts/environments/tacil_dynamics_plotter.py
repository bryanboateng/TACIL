from typing import Protocol

import numpy as np
import wandb
from numpy import typing as npt


class TacilDynamicsPlotter(Protocol):
    def __call__(
        self,
        *,
        member_trajectories: npt.NDArray[
            np.floating
        ],  # (member_count, step_count, state_dimensionality)
        true_trajectory: npt.NDArray[np.floating],  # (step_count, state_dimensionality)
        episode_index: int,
    ) -> dict[str, wandb.Image]: ...
