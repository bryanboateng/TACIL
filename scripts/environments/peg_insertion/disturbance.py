import gymnasium
import numpy as np
import sapien
from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv
from numpy import typing as npt

_FIRST_DISTURBANCE_STEP = 20
_DISTURBANCE_PERIOD = 30

_DEGENERATE_LENGTH = 1e-9


class DisturbanceWrapper(gymnasium.Wrapper):
    def __init__(self, *, environment: gymnasium.Env, magnitude: float) -> None:
        super().__init__(env=environment)

        task = environment.unwrapped
        if not isinstance(task, PegInsertionSideEnv):
            raise TypeError(f"Expected PegInsertionSideEnv, got {type(task).__name__}")

        self._task = task
        self._magnitude = magnitude
        self._time_step = 0

    @property
    def is_pushing(self) -> bool:
        return self._is_disturbance_step(time_step=self._time_step - 1)

    def reset(self, **kwargs) -> tuple:
        self._time_step = 0
        return self.env.reset(**kwargs)

    def step(self, action: npt.NDArray[np.float32]) -> tuple:
        if self._is_disturbance_step(time_step=self._time_step):
            self._apply_impulse()

        self._time_step += 1
        return self.env.step(action)

    # ---- Private --------------------------------------------

    def _is_disturbance_step(self, *, time_step: int) -> bool:
        return (
            time_step >= _FIRST_DISTURBANCE_STEP
            and (time_step - _FIRST_DISTURBANCE_STEP) % _DISTURBANCE_PERIOD == 0
        )

    def _apply_impulse(self) -> None:
        body = _find_rigid_body(task=self._task)

        push_direction = _calculate_push_direction(
            peg_head_position=_read_vector(values=self._task.peg_head_pose.p, length=3),
            hole_position=_read_vector(values=self._task.box_hole_pose.p, length=3),
            hole_quaternion=_read_vector(values=self._task.box_hole_pose.q, length=4),
        )  # (3,)

        body.set_linear_velocity(
            np.asarray(a=body.get_linear_velocity(), dtype=np.float64)
            + (self._magnitude / float(body.mass)) * push_direction
        )


def _read_vector(*, values, length: int) -> npt.NDArray[np.float64]:  # (length,)
    return np.asarray(a=values.cpu(), dtype=np.float64).reshape(-1)[:length]


def _find_rigid_body(
    *, task: PegInsertionSideEnv
) -> sapien.physx.PhysxRigidDynamicComponent:
    body = task.peg._objs[0].find_component_by_type(
        sapien.physx.PhysxRigidDynamicComponent
    )
    if body is None:
        raise TypeError("The peg carries no movable rigid body")
    return body


def _calculate_push_direction(
    *,
    peg_head_position: npt.NDArray[np.float64],  # (3,)
    hole_position: npt.NDArray[np.float64],  # (3,)
    hole_quaternion: npt.NDArray[np.float64],  # (4,)
) -> npt.NDArray[np.float64]:  # (3,)
    hole_axis = _rotate_unit_x(quaternion=hole_quaternion)  # (3,)

    lateral_direction = np.array(object=[-hole_axis[1], hole_axis[0], 0.0])  # (3,)
    length = float(np.linalg.norm(lateral_direction))
    if length < _DEGENERATE_LENGTH:
        return np.array(object=[0.0, 1.0, 0.0])
    lateral_direction = lateral_direction / length

    lateral_offset = float(
        np.dot(a=peg_head_position - hole_position, b=lateral_direction)
    )
    return lateral_direction if lateral_offset >= 0.0 else -lateral_direction


def _rotate_unit_x(
    *,
    quaternion: npt.NDArray[np.float64],  # (4,)
) -> npt.NDArray[np.float64]:  # (3,)
    real, first, second, third = quaternion
    return np.array(
        object=[
            1.0 - 2.0 * (second * second + third * third),
            2.0 * (first * second + real * third),
            2.0 * (first * third - real * second),
        ]
    )
