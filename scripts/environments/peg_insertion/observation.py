import numpy as np
from numpy import typing as npt

RAW_DIMENSIONALITY = 43

_RAW_JOINT_ANGLE_SLICE = slice(0, 9)
_RAW_JOINT_VELOCITY_SLICE = slice(9, 18)
_RAW_GRIPPER_POSITION_SLICE = slice(18, 21)
_RAW_GRIPPER_ORIENTATION_SLICE = slice(21, 25)
_RAW_PEG_POSITION_SLICE = slice(25, 28)
_RAW_PEG_ORIENTATION_SLICE = slice(28, 32)
_RAW_PEG_HALF_SIZE_SLICE = slice(32, 35)
_RAW_HOLE_POSITION_SLICE = slice(35, 38)
_RAW_HOLE_ORIENTATION_SLICE = slice(38, 42)
_RAW_HOLE_RADIUS_SLICE = slice(42, 43)

JOINT_ANGLE_SLICE = slice(0, 9)
JOINT_VELOCITY_SLICE = slice(9, 18)
GRIPPER_POSITION_SLICE = slice(18, 21)
PEG_POSITION_SLICE = slice(21, 24)
PEG_ORIENTATION_INDEX = 24
PEG_HALF_SIZE_SLICE = slice(25, 28)
HOLE_POSITION_SLICE = slice(28, 31)
HOLE_ORIENTATION_INDEX = 31
HOLE_RADIUS_INDEX = 32

STATE_DIMENSIONALITY = 33

ARM_JOINT_COUNT = 7
ACTION_DIMENSIONALITY = 8

ANGLE_INDICES = (PEG_ORIENTATION_INDEX, HOLE_ORIENTATION_INDEX)

_FULL_TURN = 2.0 * np.pi


def convert_observations(
    *,
    raw_observation_batch: npt.NDArray[np.float32],  # (batch_size, 43)
) -> npt.NDArray[np.float32]:  # (batch_size, state_dimensionality)
    if raw_observation_batch.shape[-1] != RAW_DIMENSIONALITY:
        raise ValueError(
            f"Expected {RAW_DIMENSIONALITY} raw entries, "
            f"got {raw_observation_batch.shape[-1]}"
        )

    return np.concatenate(
        [
            raw_observation_batch[:, _RAW_JOINT_ANGLE_SLICE],
            raw_observation_batch[:, _RAW_JOINT_VELOCITY_SLICE],
            raw_observation_batch[:, _RAW_GRIPPER_POSITION_SLICE],
            raw_observation_batch[:, _RAW_PEG_POSITION_SLICE],
            calculate_yaw(
                quaternion_batch=raw_observation_batch[:, _RAW_PEG_ORIENTATION_SLICE]
            )[:, None],
            raw_observation_batch[:, _RAW_PEG_HALF_SIZE_SLICE],
            raw_observation_batch[:, _RAW_HOLE_POSITION_SLICE],
            calculate_yaw(
                quaternion_batch=raw_observation_batch[:, _RAW_HOLE_ORIENTATION_SLICE]
            )[:, None],
            raw_observation_batch[:, _RAW_HOLE_RADIUS_SLICE],
        ],
        axis=-1,
        dtype=np.float32,
    )


def calculate_yaw(
    *,
    quaternion_batch: npt.NDArray[np.floating],  # (batch_size, 4)
) -> npt.NDArray[np.floating]:  # (batch_size,)
    real_batch = quaternion_batch[:, 0]
    first_batch = quaternion_batch[:, 1]
    second_batch = quaternion_batch[:, 2]
    third_batch = quaternion_batch[:, 3]
    return (
        np.arctan2(
            2.0 * (first_batch * second_batch + real_batch * third_batch),
            1.0 - 2.0 * (second_batch * second_batch + third_batch * third_batch),
        )
        % _FULL_TURN
    )
