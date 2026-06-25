import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import numpy.typing as npt
import pymunk
from gym_pusht.envs import PushTEnv
from gymnasium.spaces import Box


@dataclass(frozen=True)
class BlockShove:
    """A sustained external force on the block over a short window, then nominal dynamics.

    Models an invisible second pusher. The force is added on every physics substep of
    the window with the env's own damping left untouched, so the block moves while it is
    pushed and -- like every object in PushT's damped world -- stops almost immediately
    once the push ends. The force must be applied per substep (not as a one-shot
    velocity kick) because space damping is 0.0: any standalone velocity is annihilated
    after a single substep, which just teleports the block and looks like a glitch.

    At unit mass the block travels roughly ``force * 0.001 * duration`` units.
    """

    force: (
        float  # external force magnitude added to the block each substep of the window
    )
    step: int = 100  # control step at which the push starts
    duration: int = 10  # number of control steps the push lasts (each is 10 substeps)


@dataclass(frozen=True)
class ActionNoise:
    """Gaussian noise added to every executed action (actuator error)."""

    std: float  # standard deviation in action units


Disturbance = BlockShove | ActionNoise


class StepFunction(Protocol):
    def __call__(self, action: npt.NDArray[np.float64]) -> tuple: ...


class StepFunctionWrapper(Protocol):
    def __call__(self, step_function: StepFunction) -> StepFunction: ...


@dataclass
class DisturberState:
    time_step: int = 0
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    push_force: pymunk.Vec2d = field(default_factory=lambda: pymunk.Vec2d(0.0, 0.0))


def describe(disturbance: Disturbance | None) -> str:
    """Human-readable label of the disturbance type and its value, for on-screen overlays."""
    match disturbance:
        case None:
            return "Clean"
        case BlockShove(force=force):
            return f"Block shove  {force:g}"
        case ActionNoise(std=std):
            return f"Action noise  {std:g}"


def setup_disturber(*, environment: PushTEnv, disturbance: Disturbance) -> None:
    state = DisturberState()
    match disturbance:
        case BlockShove():
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
    config: BlockShove, state: DisturberState, environment: PushTEnv
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
    block_position: npt.NDArray[np.float64] = np.array(environment.block.position)
    angle_away_from_goal = _calculate_direction_away_from_goal(
        block_position=block_position, goal_position=np.array(environment.goal_pose[:2])
    )
    return _calculate_direction_deflected_away_from_agent(
        angle_away_from_goal=angle_away_from_goal,
        block_position=block_position,
        agent_position=np.array(environment.agent.position),
    )


def _calculate_direction_away_from_goal(
    *, block_position: npt.NDArray[np.float64], goal_position: npt.NDArray[np.float64]
) -> float:
    displacement = block_position - goal_position
    distance = float(np.linalg.norm(displacement))
    if distance > 1e-6:
        return math.atan2(float(displacement[1]), float(displacement[0]))
    return 0.0


def _calculate_direction_deflected_away_from_agent(
    *,
    angle_away_from_goal: float,
    block_position: npt.NDArray[np.float64],
    agent_position: npt.NDArray[np.float64],
) -> float:
    direction = np.array(
        [math.cos(angle_away_from_goal), math.sin(angle_away_from_goal)]
    )
    to_agent = agent_position - block_position
    agent_distance = float(np.linalg.norm(to_agent))
    if agent_distance <= 1e-6:
        return angle_away_from_goal
    to_agent = to_agent / agent_distance

    component_toward_agent = float(np.dot(direction, to_agent))
    if component_toward_agent <= 0:
        # Already pushing away from the agent; nothing to avoid.
        return angle_away_from_goal

    deflected = direction - component_toward_agent * to_agent
    deflected_norm = float(np.linalg.norm(deflected))
    if deflected_norm > 1e-6:
        deflected = deflected / deflected_norm
        return math.atan2(float(deflected[1]), float(deflected[0]))
    # direction pointed straight at the agent: slide perpendicular instead.
    return math.atan2(float(-to_agent[1]), float(to_agent[0]))
