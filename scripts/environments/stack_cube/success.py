import numpy as np
from numpy import typing as npt

from scripts.environments.stack_cube.observation import (
    CUBE_A_ANGULAR_VELOCITY_SLICE,
    CUBE_A_LINEAR_VELOCITY_SLICE,
    CUBE_A_POSITION_SLICE,
    CUBE_B_POSITION_SLICE,
    CUBE_HALF_SIZE,
    GRIPPER_OPENING_INDEX,
)

_PLACEMENT_TOLERANCE = 0.005
_STATIC_LINEAR_SPEED_LIMIT = 1e-2
_STATIC_ANGULAR_SPEED_LIMIT = 0.5
_RELEASED_GRIPPER_OPENING = 0.021


def calculate_success(
    *,
    state_batch: npt.NDArray[np.float32],  # (batch_size, STATE_DIMENSIONALITY)
) -> npt.NDArray[np.bool_]:  # (batch_size,)
    offset_batch = calculate_stacking_offset(state_batch=state_batch)  # (batch_size, 3)

    stacked_batch = (
        np.linalg.norm(offset_batch[:, :2], axis=-1)
        <= np.linalg.norm([CUBE_HALF_SIZE, CUBE_HALF_SIZE]) + _PLACEMENT_TOLERANCE
    ) & (np.abs(offset_batch[:, 2] - 2.0 * CUBE_HALF_SIZE) <= _PLACEMENT_TOLERANCE)

    static_batch = (
        np.linalg.norm(state_batch[:, CUBE_A_LINEAR_VELOCITY_SLICE], axis=-1)
        <= _STATIC_LINEAR_SPEED_LIMIT
    ) & (
        np.linalg.norm(state_batch[:, CUBE_A_ANGULAR_VELOCITY_SLICE], axis=-1)
        <= _STATIC_ANGULAR_SPEED_LIMIT
    )

    released_batch = state_batch[:, GRIPPER_OPENING_INDEX] >= _RELEASED_GRIPPER_OPENING

    return stacked_batch & static_batch & released_batch


def calculate_stacking_offset(
    *,
    state_batch: npt.NDArray[np.float32],  # (batch_size, STATE_DIMENSIONALITY)
) -> npt.NDArray[np.float64]:  # (batch_size, 3)
    return state_batch[:, CUBE_A_POSITION_SLICE].astype(np.float64) - state_batch[
        :, CUBE_B_POSITION_SLICE
    ].astype(np.float64)
