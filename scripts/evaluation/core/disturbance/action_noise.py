from dataclasses import dataclass

import numpy as np
from gym_pusht.envs import PushTEnv
from gymnasium.spaces import Box
from numpy import typing as npt


@dataclass(frozen=True)
class ActionNoise:
    std: float


def setup_action_noise(*, environment: PushTEnv, config: ActionNoise) -> None:
    action_space = environment.action_space
    if not isinstance(action_space, Box):
        raise TypeError(f"Expected Box action space, got {type(action_space).__name__}")
    action_low: npt.NDArray[np.float64] = action_space.low
    action_high: npt.NDArray[np.float64] = action_space.high

    step_function = environment.step

    def disturbed_step(
        action: npt.NDArray[np.float64],  # (action_dimensionality,)
    ) -> tuple:
        # Draw from the env's own generator so the noise is reproducible for a given
        # reset seed without tracking a separate rng. PushT only consumes np_random at
        # reset (initial pose), so sampling here doesn't perturb the task within an episode.
        noise: npt.NDArray[np.float64] = environment.np_random.normal(
            loc=0.0, scale=config.std, size=action.shape
        )  # (action_dimensionality,)

        noisy_action: npt.NDArray[np.float64] = np.clip(
            a=action + noise, a_min=action_low, a_max=action_high
        )  # (action_dimensionality,)

        return step_function(action=noisy_action)

    environment.step = disturbed_step  # ty:ignore[invalid-assignment]
