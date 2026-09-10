import gymnasium
import numpy as np
import sapien
from numpy import typing as npt

# The first disturbance lands once the policy has started to move, and further
# ones follow at a spacing that fits a few into one episode.
_FIRST_DISTURBANCE_STEP = 10
_DISTURBANCE_PERIOD = 20


class DisturbanceWrapper(gymnasium.Wrapper):
    """Kicks the manipulated object sideways, away from its goal.

    The kick is an impulse, so it changes the object's speed the way a real
    knock would. Nothing is teleported.
    """

    def __init__(
        self, *, environment: gymnasium.Env, object_name: str, magnitude: float
    ) -> None:
        super().__init__(env=environment)
        self._object_name = object_name
        self._magnitude = magnitude
        self._time_step = 0

    def reset(self, **kwargs) -> tuple:
        self._time_step = 0
        return self.env.reset(**kwargs)

    def step(self, action: npt.NDArray[np.float32]) -> tuple:
        if (
            self._time_step >= _FIRST_DISTURBANCE_STEP
            and (self._time_step - _FIRST_DISTURBANCE_STEP) % _DISTURBANCE_PERIOD == 0
        ):
            self._apply_impulse()

        self._time_step += 1
        return self.env.step(action)

    # ---- Private --------------------------------------------

    def _apply_impulse(self) -> None:
        body = _find_rigid_body(environment=self.env, object_name=self._object_name)

        object_position = np.asarray(a=body.get_pose().p, dtype=np.float64)  # (3,)
        goal_position = _find_goal_position(environment=self.env)  # (3,)

        velocity_change = (
            self._magnitude / float(body.mass)
        ) * _calculate_push_direction(
            object_position=object_position, goal_position=goal_position
        )  # (3,)

        body.set_linear_velocity(
            np.asarray(a=body.get_linear_velocity(), dtype=np.float64) + velocity_change
        )


def _find_rigid_body(
    *, environment: gymnasium.Env, object_name: str
) -> sapien.physx.PhysxRigidDynamicComponent:
    actor = getattr(environment.unwrapped, object_name)
    body = actor._objs[0].find_component_by_type(
        sapien.physx.PhysxRigidDynamicComponent
    )
    if body is None:
        raise TypeError(f"{object_name!r} carries no movable rigid body")
    return body


def _find_goal_position(*, environment: gymnasium.Env) -> npt.NDArray[np.float64]:
    goal_region = getattr(environment.unwrapped, "goal_region")
    return np.asarray(a=goal_region.pose.p, dtype=np.float64).reshape(-1)[:3]


def _calculate_push_direction(
    *,
    object_position: npt.NDArray[np.float64],  # (3,)
    goal_position: npt.NDArray[np.float64],  # (3,)
) -> npt.NDArray[np.float64]:  # (3,)
    """Point the kick sideways to the goal, so it never helps the object in."""
    object_to_goal_vector = goal_position - object_position
    object_to_goal_vector[2] = 0.0

    sideways_vector = np.array(
        object=[-object_to_goal_vector[1], object_to_goal_vector[0], 0.0]
    )  # (3,)

    length = float(np.linalg.norm(sideways_vector))
    if length < 1e-9:
        # The object sits on the goal, so there is no sideways direction.
        return np.array(object=[1.0, 0.0, 0.0])

    return sideways_vector / length
