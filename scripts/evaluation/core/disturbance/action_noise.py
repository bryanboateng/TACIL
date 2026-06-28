from dataclasses import dataclass

import numpy as np
from gym_pusht.envs import PushTEnv
from gymnasium.spaces import Box
from numpy import typing as npt

from scripts.evaluation.core.disturbance.disturbance import (
    DisturberState,
    StepFunction,
    StepFunctionWrapper,
)


@dataclass(frozen=True)
class ActionNoise:
    std: float


def _wrap_step_function_with_action_noise(
    config: ActionNoise, state: DisturberState, environment: PushTEnv
) -> StepFunctionWrapper:
    action_space = environment.action_space
    if not isinstance(action_space, Box):
        raise TypeError(f"Expected Box action space, got {type(action_space).__name__}")
    action_low: npt.NDArray[np.float64] = action_space.low
    action_high: npt.NDArray[np.float64] = action_space.high

    def wrap(step_function: StepFunction) -> StepFunction:
        def disturbed_step(
            action: npt.NDArray[np.float64],  # (action_dimensionality,)
        ) -> tuple:
            noise: npt.NDArray[np.float64] = state.rng.normal(
                loc=0.0, scale=config.std, size=action.shape
            )  # (action_dimensionality,)

            noisy_action: npt.NDArray[np.float64] = np.clip(
                a=action + noise, a_min=action_low, a_max=action_high
            )  # (action_dimensionality,)

            state.time_step += 1
            return step_function(action=noisy_action)

        return disturbed_step

    return wrap
