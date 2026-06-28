import math
from dataclasses import dataclass

import pymunk
from gym_pusht.envs import PushTEnv
from numpy import typing as npt


@dataclass(frozen=True)
class BlockPush:
    force: float
    step: int
    duration: int


def setup_block_push(*, environment: PushTEnv, config: BlockPush) -> None:
    # The push window and the force vector are the only state this needs, and both are
    # private to one episode, so they live as closure variables here rather than in a
    # shared struct. The step counter is zeroed by the reset hook below.
    time_step = 0
    push_force = pymunk.Vec2d(0.0, 0.0)

    def push_velocity(
        body: pymunk.Body, gravity: pymunk.Vec2d, damping: float, dt: float
    ) -> None:
        # Add the external shove force, then run the block's normal velocity integration
        # with the env's real damping untouched. This func is only installed during the
        # push window, so it always pushes while present. It runs on every substep --
        # setting body.force from outside env.step would only reach the first of its 10.
        body.apply_force_at_world_point(force=push_force, point=body.position)
        pymunk.Body.update_velocity(body=body, gravity=gravity, damping=damping, dt=dt)

    step_function = environment.step

    def disturbed_step(action: npt.NDArray) -> tuple:
        nonlocal time_step, push_force

        if time_step == config.step:
            angle = _calculate_push_direction(environment=environment)
            push_force = pymunk.Vec2d(math.cos(angle), math.sin(angle)) * config.force
            environment.block.velocity_func = push_velocity
        elif time_step == config.step + config.duration:
            environment.block.velocity_func = pymunk.Body.update_velocity

        time_step += 1
        return step_function(action=action)

    reset_function = environment.reset

    def disturbed_reset(**kwargs) -> tuple:
        nonlocal time_step
        result = reset_function(**kwargs)
        time_step = 0
        return result

    environment.step = disturbed_step  # ty:ignore[invalid-assignment]
    environment.reset = disturbed_reset  # ty:ignore[invalid-assignment]


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
