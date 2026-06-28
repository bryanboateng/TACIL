from dataclasses import dataclass

import gymnasium
import numpy as np
from gymnasium.spaces import Box
from numpy import typing as npt


@dataclass(frozen=True)
class Config:
    noise_std: float


class ActionNoiseWrapper(gymnasium.ActionWrapper):
    def __init__(self, environment: gymnasium.Env, config: Config) -> None:
        super().__init__(env=environment)

        action_space = self.action_space
        if not isinstance(action_space, Box):
            raise TypeError(
                f"Expected Box action space, got {type(action_space).__name__}"
            )

        self._noise_std = config.noise_std
        self._action_low: npt.NDArray[np.float64] = action_space.low
        self._action_high: npt.NDArray[np.float64] = action_space.high

    def action(
        self,
        action: npt.NDArray[np.float64],  # (action_dimensionality,)
    ) -> npt.NDArray[np.float64]:
        noise: npt.NDArray[np.float64] = self.np_random.normal(
            loc=0.0, scale=self._noise_std, size=action.shape
        )  # (action_dimensionality,)
        return np.clip(
            a=action + noise, a_min=self._action_low, a_max=self._action_high
        )  # (action_dimensionality,)
