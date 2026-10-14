import math

import gymnasium
import numpy as np
import torch
from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
from mani_skill.utils.structs import Pose
from numpy import typing as npt

from scripts.environments.stack_cube.observation import CUBE_HALF_SIZE

_LIFTED_HEIGHT = 2.0 * CUBE_HALF_SIZE
_ABOVE_BASE_RADIUS = math.hypot(CUBE_HALF_SIZE, CUBE_HALF_SIZE) + 0.005


class DisturbanceWrapper(gymnasium.Wrapper):
    def __init__(self, *, environment: gymnasium.Env, shift: float) -> None:
        super().__init__(env=environment)

        task = environment.unwrapped
        if not isinstance(task, StackCubeEnv):
            raise TypeError(f"Expected StackCubeEnv, got {type(task).__name__}")

        self._task = task
        self._shift = shift
        self._has_shifted = False

    def reset(self, **kwargs) -> tuple:
        self._has_shifted = False
        return self.env.reset(**kwargs)

    def step(self, action: npt.NDArray[np.float32]) -> tuple:
        if not self._has_shifted and self._is_cube_carried_above_base():
            self._shift_base_cube()
            self._has_shifted = True

        return self.env.step(action)

    # ---- Private --------------------------------------------

    def _is_cube_carried_above_base(self) -> bool:
        offset = (self._task.cubeA.pose.p - self._task.cubeB.pose.p).cpu().reshape(-1)
        lifted = float(self._task.cubeA.pose.p.cpu().reshape(-1)[2]) > _LIFTED_HEIGHT
        return lifted and float(torch.linalg.norm(offset[:2])) <= _ABOVE_BASE_RADIUS

    def _shift_base_cube(self) -> None:
        offset = (self._task.cubeA.pose.p - self._task.cubeB.pose.p).cpu().reshape(-1)
        lateral_sign = -1.0 if float(offset[0]) >= 0.0 else 1.0

        self._task.cubeB.set_pose(
            Pose.create_from_pq(
                p=torch.tensor(data=[lateral_sign * self._shift, 0.0, 0.0])
            )
            * self._task.cubeB.pose
        )
