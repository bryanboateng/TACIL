import math
from dataclasses import dataclass

import gymnasium
import pymunk
from gym_pusht.envs import PushTEnv
from numpy import typing as npt
from pymunk import Body


@dataclass(frozen=True)
class Config:
    magnitude: float
    window: range


class BlockPushWrapper(gymnasium.Wrapper):
    def __init__(self, environment: gymnasium.Env, config: Config) -> None:
        super().__init__(env=environment)

        push_t_environment = environment.unwrapped
        if not isinstance(push_t_environment, PushTEnv):
            raise TypeError(
                f"Expected PushTEnv, got {type(push_t_environment).__name__}"
            )

        self._config = config
        self._push_t_environment = push_t_environment
        self._time_step = 0

    def reset(self, **kwargs) -> tuple:
        self._time_step = 0
        return self.env.reset(**kwargs)

    def step(self, action: npt.NDArray) -> tuple:
        if self._time_step in self._config.window:
            angle = _calculate_push_angle(environment=self._push_t_environment)
            impulse = (
                pymunk.Vec2d(x=math.cos(angle), y=math.sin(angle))
                * self._config.magnitude
            )
            block: Body = self._push_t_environment.block
            block.apply_impulse_at_world_point(
                impulse=impulse, point=block.local_to_world(v=block.center_of_gravity)
            )

        self._time_step += 1

        return self.env.step(action=action)


def _calculate_push_angle(*, environment: PushTEnv) -> float:
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
