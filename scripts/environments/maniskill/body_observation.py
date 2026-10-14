import numpy as np
from mani_skill.utils.structs import Actor, Link, Pose
from numpy import typing as npt


def extract_pose(*, pose: Pose) -> npt.NDArray[np.float32]:  # (9,)
    position = pose.p.cpu().numpy().reshape(3)  # (3,)
    quaternion = pose.q.cpu().numpy().reshape(4)  # (4,) w,x,y,z

    return np.concatenate(
        [position, quaternion_to_6d(quaternion=quaternion)], dtype=np.float32
    )


def extract_pose_and_velocity(
    *, body: Actor | Link
) -> npt.NDArray[np.float32]:  # (15,)
    linear_velocity = body.get_linear_velocity().cpu().numpy().reshape(3)  # (3,)
    angular_velocity = body.get_angular_velocity().cpu().numpy().reshape(3)  # (3,)

    return np.concatenate(
        [extract_pose(pose=body.pose), linear_velocity, angular_velocity],
        dtype=np.float32,
    )


def quaternion_to_6d(
    *,
    quaternion: npt.NDArray[np.floating],  # (4,) w,x,y,z
) -> npt.NDArray[np.floating]:  # (6,)
    w, x, y, z = quaternion
    return np.array(
        object=[
            1 - 2 * (y * y + z * z),
            2 * (x * y + w * z),
            2 * (x * z - w * y),
            2 * (x * y - w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z + w * x),
        ]
    )
