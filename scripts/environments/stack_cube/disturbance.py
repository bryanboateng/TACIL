import gymnasium
import numpy as np
import torch
from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
from mani_skill.utils.structs import Pose
from numpy import typing as npt

_FIRST_DISTURBANCE_STEP = 20
_DISTURBANCE_PERIOD = 100


class DisturbanceWrapper(gymnasium.Wrapper):
    def __init__(self, *, environment: gymnasium.Env, shift: float) -> None:
        super().__init__(env=environment)

        task = environment.unwrapped
        if not isinstance(task, StackCubeEnv):
            raise TypeError(f"Expected StackCubeEnv, got {type(task).__name__}")

        self._task = task
        self._shift = shift
        self._time_step = 0

    def reset(self, **kwargs) -> tuple:
        self._time_step = 0
        return self.env.reset(**kwargs)

    def step(self, action: npt.NDArray[np.float32]) -> tuple:
        steps_since_first = self._time_step - _FIRST_DISTURBANCE_STEP
        if steps_since_first >= 0 and steps_since_first % _DISTURBANCE_PERIOD == 0:
            self._shift_base_cube()

        self._time_step += 1
        return self.env.step(action)

    # ---- Private --------------------------------------------

    def _shift_base_cube(self) -> None:
        offset = (self._task.cubeA.pose.p - self._task.cubeB.pose.p).cpu().reshape(-1)
        lateral_sign = -1.0 if float(offset[0]) >= 0.0 else 1.0

        self._task.cubeB.set_pose(
            Pose.create_from_pq(
                p=torch.tensor(data=[lateral_sign * self._shift, 0.0, 0.0])
            )
            * self._task.cubeB.pose
        )
