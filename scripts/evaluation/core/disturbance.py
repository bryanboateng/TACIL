import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import numpy.typing as npt
import pymunk
from gym_pusht.envs import PushTEnv
from gymnasium.spaces import Box


@dataclass(frozen=True)
class BlockPush:
    force: float
    step: int
    duration: int


@dataclass(frozen=True)
class ActionNoise:
    std: float


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


def _wrap_step_function_with_block_shove(
    config: BlockPush, state: DisturberState, environment: PushTEnv
) -> StepFunctionWrapper:
    def push_velocity(
        body: pymunk.Body, gravity: pymunk.Vec2d, damping: float, dt: float
    ) -> None:
        # Add the external shove force, then run the block's normal velocity integration
        # with the env's real damping untouched. This func is only installed during the
        # push window, so it always pushes while present. It runs on every substep --
        # setting body.force from outside env.step would only reach the first of its 10.
        body.apply_force_at_world_point(state.push_force, body.position)
        pymunk.Body.update_velocity(body, gravity, damping, dt)

    def wrap(step_function: StepFunction) -> StepFunction:
        def disturbed_step(action: npt.NDArray[np.float64]) -> tuple:
            # Aim the shove away from the goal (deflected off the agent) at the start step
            # and install the per-substep force hook; remove it once the window closes so
            # the rest of the episode runs clean and tests recovery.
            if state.time_step == config.step:
                angle = _calculate_push_direction(environment=environment)
                state.push_force = (
                    pymunk.Vec2d(math.cos(angle), math.sin(angle)) * config.force
                )
                environment.block.velocity_func = push_velocity
            elif state.time_step == config.step + config.duration:
                environment.block.velocity_func = pymunk.Body.update_velocity
            state.time_step += 1
            return step_function(action=action)

        return disturbed_step

    return wrap


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


def _patch_reset(*, environment: PushTEnv, state: DisturberState) -> None:
    original_reset = environment.reset

    def new_reset(**kwargs):
        result = original_reset(**kwargs)
        state.time_step = 0
        # Reseed per episode so the disturbance is reproducible for a given seed.
        state.rng = np.random.default_rng(kwargs.get("seed"))
        return result

    environment.reset = new_reset  # ty:ignore[invalid-assignment]


def _calculate_push_direction(*, environment: PushTEnv) -> float:
    block_position = environment.block.position
    angle_away_from_goal = _calculate_angle_away_from_goal(
        block_position=block_position,
        goal_position=pymunk.Vec2d(*environment.goal_pose[:2]),
    )
    return _calculate_angle_deflected_away_from_agent(
        angle_away_from_goal=angle_away_from_goal,
        block_position=block_position,
        agent_position=environment.agent.position,
    )


def _calculate_angle_away_from_goal(
    *, block_position: pymunk.Vec2d, goal_position: pymunk.Vec2d
) -> float:

    # noinspection PyTypeChecker
    # Vec2d.__sub__ expects tuple[float, float], but doesn't recognize Vec2d as compatible since it's a NamedTuple subclass
    displacement = block_position - goal_position

    if displacement.length > 1e-6:
        return displacement.angle
    return 0.0


def _calculate_angle_deflected_away_from_agent(
    *,
    angle_away_from_goal: float,
    block_position: pymunk.Vec2d,
    agent_position: pymunk.Vec2d,
) -> float:
    direction = pymunk.Vec2d(
        math.cos(angle_away_from_goal), math.sin(angle_away_from_goal)
    )
    to_agent = agent_position - block_position
    if to_agent.length <= 1e-6:
        return angle_away_from_goal

    to_agent = to_agent.normalized()

    component_toward_agent = direction.dot(to_agent)
    if component_toward_agent <= 0:
        # Already pushing away from the agent; nothing to avoid.
        return angle_away_from_goal

    deflected = direction - component_toward_agent * to_agent
    if deflected.length > 1e-6:
        return deflected.angle
    # direction pointed straight at the agent: slide perpendicular instead.
    return math.atan2(-to_agent.y, to_agent.x)
