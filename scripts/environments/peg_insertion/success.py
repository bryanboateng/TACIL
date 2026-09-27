import numpy as np
from numpy import typing as npt

from scripts.environments.peg_insertion.observation import (
    HOLE_ORIENTATION_SLICE,
    HOLE_POSITION_SLICE,
    HOLE_RADIUS_INDEX,
    PEG_HALF_SIZE_SLICE,
    PEG_FORWARD_AXIS_SLICE,
    PEG_POSITION_SLICE,
)

_INSERTION_DEPTH = 0.015


def calculate_success(
    *,
    state_batch: npt.NDArray[np.float32],  # (batch_size, STATE_DIMENSIONALITY)
) -> npt.NDArray[np.bool_]:  # (batch_size,)
    head_offset_batch = calculate_peg_head_offset(
        state_batch=state_batch
    )  # (batch_size, 3)
    radius_batch = state_batch[:, HOLE_RADIUS_INDEX]  # (batch_size,)

    return (
        (head_offset_batch[:, 0] >= -_INSERTION_DEPTH)
        & (np.abs(head_offset_batch[:, 1]) <= radius_batch)
        & (np.abs(head_offset_batch[:, 2]) <= radius_batch)
    )


def calculate_peg_head_offset(
    *,
    state_batch: npt.NDArray[np.float32],  # (batch_size, STATE_DIMENSIONALITY)
) -> npt.NDArray[np.float64]:  # (batch_size, 3)
    peg_forward_axis_batch = state_batch[:, PEG_FORWARD_AXIS_SLICE].astype(
        np.float64
    )  # (batch_size, 3)
    hole_rotation_batch = _create_rotations(
        orientation_batch=state_batch[:, HOLE_ORIENTATION_SLICE].astype(np.float64)
    )  # (batch_size, 3, 3)

    half_length_batch = state_batch[:, PEG_HALF_SIZE_SLICE][:, 0].astype(
        np.float64
    )  # (batch_size,)

    head_position_batch = (
        state_batch[:, PEG_POSITION_SLICE].astype(np.float64)
        + half_length_batch[:, None] * peg_forward_axis_batch
    )  # (batch_size, 3)

    return np.einsum(
        "bji,bj->bi",
        hole_rotation_batch,
        head_position_batch - state_batch[:, HOLE_POSITION_SLICE].astype(np.float64),
    )  # (batch_size, 3)


# ---- Private --------------------------------------------


def _create_rotations(
    *,
    orientation_batch: npt.NDArray[np.float64],  # (batch_size, 6)
) -> npt.NDArray[np.float64]:  # (batch_size, 3, 3)
    first_column_batch = orientation_batch[:, :3]  # (batch_size, 3)
    second_column_batch = orientation_batch[:, 3:]  # (batch_size, 3)
    third_column_batch = np.cross(
        a=first_column_batch, b=second_column_batch
    )  # (batch_size, 3)
    return np.stack(
        arrays=[first_column_batch, second_column_batch, third_column_batch], axis=-1
    )  # (batch_size, 3, 3)
