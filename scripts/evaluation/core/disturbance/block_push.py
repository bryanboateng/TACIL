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
            direction = _calculate_push_direction(environment=self._push_t_environment)
            impulse = direction * self._config.magnitude

            block: Body = self._push_t_environment.block
            center_of_gravity = block.center_of_gravity
            impulse_point = block.local_to_world(
                v=(center_of_gravity.x, center_of_gravity.y)
            )
            block.apply_impulse_at_world_point(
                impulse=(impulse.x, impulse.y), point=(impulse_point.x, impulse_point.y)
            )

        self._time_step += 1

        return self.env.step(action=action)


def _calculate_push_direction(*, environment: PushTEnv) -> pymunk.Vec2d:
    block_position: pymunk.Vec2d = environment.block.position
    goal_position = pymunk.Vec2d(*environment.goal_pose[:2])
    agent_position: pymunk.Vec2d = environment.agent.position

    direction_away_from_goal = (block_position - goal_position).normalized()
    direction_to_agent = (agent_position - block_position).normalized()

    component_toward_agent = direction_away_from_goal.dot(other=direction_to_agent)
    if component_toward_agent <= 0:
        # Already pushing away from the agent. Nothing to avoid.
        return direction_away_from_goal

    # Remove the part of the push that points at the agent, keeping only the
    # perpendicular slide.
    # If that leaves nothing (i.e. the push is aimed straight at the agent), slide sideways instead.
    deflected_direction = (
        direction_away_from_goal - component_toward_agent * direction_to_agent
    )
    if deflected_direction.length > 1e-6:
        return deflected_direction.normalized()
    return direction_to_agent.perpendicular()
