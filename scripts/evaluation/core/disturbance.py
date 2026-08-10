import gymnasium
import numpy as np
import pymunk
from gym_pusht.envs import PushTEnv
from numpy import typing as npt
from pymunk import Vec2d


class DisturbanceWrapper(gymnasium.Wrapper):
    def __init__(self, environment: gymnasium.Env, magnitude: float) -> None:
        super().__init__(env=environment)

        push_t_environment = environment.unwrapped
        if not isinstance(push_t_environment, PushTEnv):
            raise TypeError(
                f"Expected PushTEnv, got {type(push_t_environment).__name__}"
            )

        self._push_t_environment = push_t_environment
        self._magnitude = magnitude

        self._time_step = 0
        self._disturbance_force = Vec2d.zero()

    @property
    def is_pushing(self) -> bool:
        return self._disturbance_force != Vec2d.zero()

    @property
    def push_direction(self) -> Vec2d:
        return self._disturbance_force.normalized()

    def reset(self, **kwargs) -> tuple:
        self._time_step = 0
        self._disturbance_force = Vec2d.zero()

        observation = self.env.reset(**kwargs)
        self._push_t_environment.block.velocity_func = self._apply_disturbance_force
        return observation

    def step(self, action: npt.NDArray[np.float32]) -> tuple:
        phase = self._time_step % 50
        if phase == 0:
            push_direction = _calculate_push_direction(
                environment=self._push_t_environment
            )
            self._disturbance_force = push_direction * self._magnitude
        elif phase == 10:
            self._disturbance_force = Vec2d.zero()

        self._time_step += 1
        return self.env.step(action=action)

    def _apply_disturbance_force(
        self, body: pymunk.Body, gravity: Vec2d, damping: float, timestep_seconds: float
    ) -> None:
        body.force = self._disturbance_force
        pymunk.Body.update_velocity(
            body=body, gravity=gravity, damping=damping, dt=timestep_seconds
        )


# ---- Private --------------------------------------------


def _calculate_push_direction(*, environment: PushTEnv) -> Vec2d:
    block_position: Vec2d = environment.block.local_to_world(
        v=environment.block.center_of_gravity
    )
    agent_position: Vec2d = environment.agent.position

    block_to_agent_vector = block_position - agent_position

    block_to_agent_normal_vector = block_to_agent_vector.perpendicular().normalized()
    goal_to_block_vector = Vec2d(*environment.goal_pose[:2]) - block_position

    # Of the two perpendicular directions, choose the one that moves the block away from
    # the goal, so the burst is always adversarial rather than helping the
    # block into the goal.
    return (
        block_to_agent_normal_vector
        if block_to_agent_normal_vector.dot(other=goal_to_block_vector) <= 0
        else -block_to_agent_normal_vector
    )
