from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import numpy.typing as npt
import pymunk
from gym_pusht.envs import PushTEnv

from scripts.evaluation.core.disturbance.action_noise import (
    ActionNoise,
    _wrap_step_function_with_action_noise,
)
from scripts.evaluation.core.disturbance.block_push import (
    BlockPush,
    _wrap_step_function_with_block_shove,
)

Disturbance = BlockPush | ActionNoise


class StepFunction(Protocol):
    def __call__(self, action: npt.NDArray[np.float64]) -> tuple: ...


class StepFunctionWrapper(Protocol):
    def __call__(self, step_function: StepFunction) -> StepFunction: ...


@dataclass
class DisturberState:
    time_step: int = 0
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    push_force: pymunk.Vec2d = field(default_factory=lambda: pymunk.Vec2d(0.0, 0.0))


def setup_disturber(*, environment: PushTEnv, disturbance: Disturbance) -> None:
    state = DisturberState()
    match disturbance:
        case BlockPush():
            _install_step(
                environment=environment,
                make_step=_wrap_step_function_with_block_shove(
                    config=disturbance, state=state, environment=environment
                ),
            )
        case ActionNoise():
            _install_step(
                environment=environment,
                make_step=_wrap_step_function_with_action_noise(
                    config=disturbance, state=state, environment=environment
                ),
            )
    _patch_reset(environment=environment, state=state)


def _install_step(*, environment: PushTEnv, make_step: StepFunctionWrapper) -> None:
    environment.step = make_step(step_function=environment.step)  # ty:ignore[invalid-assignment]


def _patch_reset(*, environment: PushTEnv, state: DisturberState) -> None:
    original_reset = environment.reset

    def new_reset(**kwargs):
        result = original_reset(**kwargs)
        state.time_step = 0
        # Reseed per episode so the disturbance is reproducible for a given seed.
        state.rng = np.random.default_rng(kwargs.get("seed"))
        return result

    environment.reset = new_reset  # ty:ignore[invalid-assignment]
