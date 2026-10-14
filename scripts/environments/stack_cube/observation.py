import numpy as np
from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
from numpy import typing as npt

from scripts.environments.maniskill.body_observation import extract_pose_and_velocity

TCP_POSITION_SLICE = slice(0, 3)
TCP_ORIENTATION_SLICE = slice(3, 9)  # (6,) first two columns of rotation matrix
TCP_LINEAR_VELOCITY_SLICE = slice(9, 12)
TCP_ANGULAR_VELOCITY_SLICE = slice(12, 15)
GRIPPER_OPENING_INDEX = 15
CUBE_A_POSITION_SLICE = slice(16, 19)
CUBE_A_ORIENTATION_SLICE = slice(19, 25)  # (6,) 6D
CUBE_A_LINEAR_VELOCITY_SLICE = slice(25, 28)
CUBE_A_ANGULAR_VELOCITY_SLICE = slice(28, 31)
CUBE_B_POSITION_SLICE = slice(31, 34)
CUBE_B_ORIENTATION_SLICE = slice(34, 40)  # (6,) 6D
CUBE_B_LINEAR_VELOCITY_SLICE = slice(40, 43)
CUBE_B_ANGULAR_VELOCITY_SLICE = slice(43, 46)

STATE_DIMENSIONALITY = 46

CUBE_HALF_SIZE = 0.02


def extract_state_from_environment(
    *, stack_cube_env: StackCubeEnv
) -> npt.NDArray[np.float32]:  # (STATE_DIMENSIONALITY,)
    tcp_pose_and_velocity = extract_pose_and_velocity(
        body=stack_cube_env.agent.tcp
    )  # (15,)
    gripper_opening = (
        stack_cube_env.agent.robot.get_qpos().cpu().numpy().reshape(-1)[7:8]
    )  # (1,)

    cube_a_pose_and_velocity = extract_pose_and_velocity(
        body=stack_cube_env.cubeA
    )  # (15,)
    cube_b_pose_and_velocity = extract_pose_and_velocity(
        body=stack_cube_env.cubeB
    )  # (15,)

    state = np.concatenate(
        [
            tcp_pose_and_velocity,
            gripper_opening,
            cube_a_pose_and_velocity,
            cube_b_pose_and_velocity,
        ],
        dtype=np.float32,
    )

    if len(state) != STATE_DIMENSIONALITY:
        raise ValueError(
            f"Wrong state dimensionality. Expected {STATE_DIMENSIONALITY}, got {len(state)}"
        )

    return state
