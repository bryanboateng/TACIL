import numpy as np
from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv
from numpy import typing as npt

from scripts.environments.maniskill.body_observation import (
    extract_pose,
    extract_pose_and_velocity,
)

TCP_POSITION_SLICE = slice(0, 3)
TCP_ORIENTATION_SLICE = slice(3, 9)  # (6,) first two columns of rotation matrix
TCP_LINEAR_VELOCITY_SLICE = slice(9, 12)
TCP_ANGULAR_VELOCITY_SLICE = slice(12, 15)
GRIPPER_OPENING_INDEX = 15
PEG_POSITION_SLICE = slice(16, 19)
PEG_ORIENTATION_SLICE = slice(19, 25)  # (6,) 6D
PEG_FORWARD_AXIS_SLICE = slice(19, 22)  # (3,) first column of the rotation matrix
PEG_LINEAR_VELOCITY_SLICE = slice(25, 28)
PEG_ANGULAR_VELOCITY_SLICE = slice(28, 31)
PEG_HALF_SIZE_SLICE = slice(31, 34)
HOLE_POSITION_SLICE = slice(34, 37)
HOLE_ORIENTATION_SLICE = slice(37, 43)  # (6,) 6D
HOLE_RADIUS_INDEX = 43

STATE_DIMENSIONALITY = 44

ACTION_DIMENSIONALITY = 7
ARM_ACTION_COUNT = 6

ANGLE_INDICES = ()


def extract_state_from_environment(
    *, peg_insertion_side_env: PegInsertionSideEnv
) -> npt.NDArray[np.float32]:  # (STATE_DIMENSIONALITY,)
    tcp_pose_and_velocity = extract_pose_and_velocity(
        body=peg_insertion_side_env.agent.tcp
    )  # (15,)
    gripper_opening = (
        peg_insertion_side_env.agent.robot.get_qpos().cpu().numpy().reshape(-1)[7:8]
    )  # (1,)

    peg_pose_and_velocity = extract_pose_and_velocity(
        body=peg_insertion_side_env.peg
    )  # (15,)
    peg_half_size = (
        peg_insertion_side_env.peg_half_sizes.cpu().numpy().reshape(3)
    )  # (3,)

    hole_pose = extract_pose(pose=peg_insertion_side_env.box_hole_pose)  # (9,)
    hole_radius = peg_insertion_side_env.box_hole_radii.cpu().numpy().reshape(1)  # (1,)

    state = np.concatenate(
        [
            tcp_pose_and_velocity,
            gripper_opening,
            peg_pose_and_velocity,
            peg_half_size,
            hole_pose,
            hole_radius,
        ],
        dtype=np.float32,
    )

    if len(state) != STATE_DIMENSIONALITY:
        raise ValueError(
            f"Wrong state dimensionality. Expected {STATE_DIMENSIONALITY}, got {len(state)}"
        )

    return state
