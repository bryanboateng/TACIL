from dataclasses import dataclass

import gymnasium
import numpy as np
from gymnasium.spaces import Box
from numpy import typing as npt


@dataclass(frozen=True)
class Config:
    # Noise std as a fraction of each action dimension's min-max range.
    noise_fraction: float


class ActionNoiseWrapper(gymnasium.ActionWrapper):
    def __init__(self, environment: gymnasium.Env, config: Config) -> None:
        super().__init__(env=environment)

        action_space = self.action_space
        if not isinstance(action_space, Box):
            raise TypeError(
                f"Expected Box action space, got {type(action_space).__name__}"
            )

        self._action_lower_bound: npt.NDArray[np.float64] = (
            action_space.low
        )  # (action_dimensionality,)

        self._action_upper_bound: npt.NDArray[np.float64] = (
            action_space.high
        )  # (action_dimensionality,)

        self._noise_std: npt.NDArray[np.float64] = config.noise_fraction * (
            self._action_upper_bound - self._action_lower_bound
        )  # (action_dimensionality,)

    def action(
        self,
        action: npt.NDArray[np.float64],  # (action_dimensionality,)
    ) -> npt.NDArray[np.float64]:
        noise: npt.NDArray[np.float64] = self.np_random.normal(
            loc=0.0, scale=self._noise_std, size=action.shape
        )  # (action_dimensionality,)
        return np.clip(
            a=action + noise,
            a_min=self._action_lower_bound,
            a_max=self._action_upper_bound,
        )  # (action_dimensionality,)
