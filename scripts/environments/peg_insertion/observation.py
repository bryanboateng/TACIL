import numpy as np
from numpy import typing as npt

TCP_POSITION_SLICE = slice(0, 3)
TCP_ORIENTATION_SLICE = slice(3, 9)  # (6,) 6D: first two columns of rotation matrix
TCP_LINEAR_VELOCITY_SLICE = slice(9, 12)
TCP_ANGULAR_VELOCITY_SLICE = slice(12, 15)
GRIPPER_OPENING_INDEX = 15
PEG_POSITION_SLICE = slice(16, 19)
PEG_ORIENTATION_SLICE = slice(19, 25)  # (6,) 6D
PEG_FORWARD_AXIS_SLICE = slice(19, 22)  # (3,) first column of the 6D rotation
PEG_LINEAR_VELOCITY_SLICE = slice(25, 28)
PEG_ANGULAR_VELOCITY_SLICE = slice(28, 31)
PEG_HALF_SIZE_SLICE = slice(31, 34)
HOLE_POSITION_SLICE = slice(34, 37)
HOLE_ORIENTATION_INDEX = 37
HOLE_RADIUS_INDEX = 38

STATE_DIMENSIONALITY = 39

ACTION_DIMENSIONALITY = 7  # (7,) 6 EE deltas + 1 gripper
ARM_ACTION_COUNT = 6

ANGLE_INDICES = (HOLE_ORIENTATION_INDEX,)

_FULL_TURN = 2.0 * np.pi


def observe_state(
    task,  # mani_skill PegInsertionSideEnv (unwrapped)
) -> npt.NDArray[np.float32]:  # (STATE_DIMENSIONALITY,)
    tcp = task.agent.tcp
    tcp_pos = tcp.pose.p.cpu().numpy().reshape(3)
    tcp_quat = tcp.pose.q.cpu().numpy().reshape(4)  # (4,) w,x,y,z
    tcp_lin_vel = tcp.get_linear_velocity().cpu().numpy().reshape(3)
    tcp_ang_vel = tcp.get_angular_velocity().cpu().numpy().reshape(3)

    gripper_opening = task.agent.robot.get_qpos().cpu().numpy().reshape(-1)[7:8]  # (1,)

    peg = task.peg
    peg_pos = peg.pose.p.cpu().numpy().reshape(3)
    peg_quat = peg.pose.q.cpu().numpy().reshape(4)  # (4,) w,x,y,z
    peg_lin_vel = peg.get_linear_velocity().cpu().numpy().reshape(3)
    peg_ang_vel = peg.get_angular_velocity().cpu().numpy().reshape(3)

    peg_half_size = task.peg_half_sizes.cpu().numpy().reshape(3)

    hole_pose = task.box_hole_pose
    hole_pos = hole_pose.p.cpu().numpy().reshape(3)
    hole_quat = hole_pose.q.cpu().numpy().reshape(4)  # (4,) w,x,y,z

    hole_radius = task.box_hole_radii.cpu().numpy().reshape(1)

    return np.concatenate(
        [
            tcp_pos,
            quaternion_to_6d(quaternion=tcp_quat),
            tcp_lin_vel,
            tcp_ang_vel,
            gripper_opening,
            peg_pos,
            quaternion_to_6d(quaternion=peg_quat),
            peg_lin_vel,
            peg_ang_vel,
            peg_half_size,
            hole_pos,
            calculate_yaw(quaternion=hole_quat),
            hole_radius,
        ],
        dtype=np.float32,
    )


def quaternion_to_6d(
    *,
    quaternion: npt.NDArray[np.floating],  # (4,) w,x,y,z
) -> npt.NDArray[np.floating]:  # (6,)
    w, x, y, z = quaternion
    return np.array(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y + w * z),
            2 * (x * z - w * y),
            2 * (x * y - w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z + w * x),
        ]
    )


def calculate_yaw(
    *,
    quaternion: npt.NDArray[np.floating],  # (4,) w,x,y,z
) -> npt.NDArray[np.floating]:  # (1,)
    w, x, y, z = quaternion
    yaw = np.arctan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (y * y + z * z))
    return np.array([yaw % _FULL_TURN])
