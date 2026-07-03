from dataclasses import dataclass

import gymnasium
from gym_pusht.envs import PushTEnv
from numpy import typing as npt
from pymunk import Body, Vec2d


@dataclass(frozen=True)
class Config:
    magnitude: float


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
        # Direction of the push applied on the most recent step, or None when none was
        # applied. Exposed so a renderer can visualize the disturbance.
        self.push_direction: Vec2d | None = None

    def reset(self, **kwargs) -> tuple:
        self._time_step = 0
        self.push_direction = None
        return self.env.reset(**kwargs)

    def step(self, action: npt.NDArray) -> tuple:
        period = 50
        duration = 10
        phase = self._time_step % period
        if phase < duration:
            # Freeze a fresh direction at the start of each burst and hold it for the
            # burst's duration. The block is quasi-static (damping zeroes its velocity
            # each step), so the force must be sustained to move it at all; freezing the
            # direction keeps the whole burst a single straight shove.
            if phase == 0:
                self.push_direction = _calculate_push_direction(
                    environment=self._push_t_environment
                )

            if self.push_direction is not None:
                block: Body = self._push_t_environment.block
                block.apply_impulse_at_world_point(
                    impulse=self.push_direction * self._config.magnitude,
                    point=block.local_to_world(v=block.center_of_gravity),
                )
        else:
            self.push_direction = None

        self._time_step += 1

        return self.env.step(action=action)


def _calculate_push_direction(*, environment: PushTEnv) -> Vec2d | None:
    block_position: Vec2d = environment.block.position
    agent_position: Vec2d = environment.agent.position
    # Push perpendicular to the agent's line of contact (agent -> block), so the block is
    # knocked sideways off the agent's push no matter where the agent currently is. The
    # agent must then reposition around the block -- an off-distribution maneuver
    # underrepresented in the expert data, which is what makes recovery hard.
    contact_axis = block_position - agent_position
    if contact_axis.length < 1e-6:
        return None
    lateral = contact_axis.perpendicular().normalized()
    # Of the two perpendicular directions, choose the one that moves the block away from
    # the goal, so the burst is always adversarial rather than occasionally helping the
    # block into the goal. Tracking the block -> goal geometry also self-corrects against
    # the one-way wall drift that a fixed side would accumulate.
    to_goal = Vec2d(*environment.goal_pose[:2]) - block_position
    return lateral if lateral.dot(to_goal) <= 0 else -lateral
