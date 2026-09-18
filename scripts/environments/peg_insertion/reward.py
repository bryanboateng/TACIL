import numpy as np
from numpy import typing as npt

from scripts.environments.peg_insertion.observation import (
    GRIPPER_POSITION_SLICE,
    HOLE_ORIENTATION_INDEX,
    HOLE_POSITION_SLICE,
    HOLE_RADIUS_INDEX,
    PEG_HALF_SIZE_SLICE,
    PEG_ORIENTATION_INDEX,
    PEG_POSITION_SLICE,
)

_INSERTION_DEPTH = 0.015

_REACHABLE_POSITION_LIMIT = 2.0


def calculate_termination(
    *,
    state_batch: npt.NDArray[np.float32],  # (batch_size, state_dimensionality)
) -> npt.NDArray[np.bool_]:  # (batch_size,)
    reachable_batch = _calculate_reachable(state_batch=state_batch)  # (batch_size,)

    termination_batch = ~reachable_batch  # (batch_size,)
    if not reachable_batch.any():
        return termination_batch

    termination_batch[reachable_batch] = calculate_success(
        state_batch=state_batch[reachable_batch]
    )
    return termination_batch


def calculate_success(
    *,
    state_batch: npt.NDArray[np.float32],  # (batch_size, state_dimensionality)
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
    state_batch: npt.NDArray[np.float32],  # (batch_size, state_dimensionality)
) -> npt.NDArray[np.float64]:  # (batch_size, 3)
    peg_rotation_batch = _create_yaw_rotations(
        yaw_batch=state_batch[:, PEG_ORIENTATION_INDEX].astype(np.float64)
    )  # (batch_size, 3, 3)
    hole_rotation_batch = _create_yaw_rotations(
        yaw_batch=state_batch[:, HOLE_ORIENTATION_INDEX].astype(np.float64)
    )  # (batch_size, 3, 3)

    half_length_batch = state_batch[:, PEG_HALF_SIZE_SLICE][:, 0].astype(
        np.float64
    )  # (batch_size,)
    head_in_peg_frame_batch = np.stack(
        arrays=[
            half_length_batch,
            np.zeros_like(a=half_length_batch),
            np.zeros_like(a=half_length_batch),
        ],
        axis=-1,
    )  # (batch_size, 3)

    head_position_batch = state_batch[:, PEG_POSITION_SLICE].astype(
        np.float64
    ) + np.einsum(
        "bij,bj->bi", peg_rotation_batch, head_in_peg_frame_batch
    )  # (batch_size, 3)

    return np.einsum(
        "bji,bj->bi",
        hole_rotation_batch,
        head_position_batch - state_batch[:, HOLE_POSITION_SLICE].astype(np.float64),
    )  # (batch_size, 3)


# ---- Private --------------------------------------------


def _calculate_reachable(
    *,
    state_batch: npt.NDArray[np.float32],  # (batch_size, state_dimensionality)
) -> npt.NDArray[np.bool_]:  # (batch_size,)
    position_batch = np.concatenate(
        [
            state_batch[:, GRIPPER_POSITION_SLICE],
            state_batch[:, PEG_POSITION_SLICE],
            state_batch[:, HOLE_POSITION_SLICE],
        ],
        axis=-1,
    )  # (batch_size, 9)

    return (
        np.isfinite(state_batch).all(axis=-1)
        & (np.abs(position_batch) < _REACHABLE_POSITION_LIMIT).all(axis=-1)
        & (state_batch[:, HOLE_RADIUS_INDEX] > 0.0)
        & (state_batch[:, PEG_HALF_SIZE_SLICE][:, 0] > 0.0)
    )


def _create_yaw_rotations(
    *,
    yaw_batch: npt.NDArray[np.float64],  # (batch_size,)
) -> npt.NDArray[np.float64]:  # (batch_size, 3, 3)
    cosine_batch = np.cos(yaw_batch)  # (batch_size,)
    sine_batch = np.sin(yaw_batch)  # (batch_size,)
    zero_batch = np.zeros_like(a=yaw_batch)  # (batch_size,)
    one_batch = np.ones_like(a=yaw_batch)  # (batch_size,)
    return np.stack(
        arrays=[
            np.stack(arrays=[cosine_batch, -sine_batch, zero_batch], axis=-1),
            np.stack(arrays=[sine_batch, cosine_batch, zero_batch], axis=-1),
            np.stack(arrays=[zero_batch, zero_batch, one_batch], axis=-1),
        ],
        axis=-2,
    )  # (batch_size, 3, 3)
