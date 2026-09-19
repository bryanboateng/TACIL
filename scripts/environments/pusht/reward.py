import numpy as np
from gym_pusht.envs.pusht import PushTEnv
from numpy import typing as npt

from scripts.shared.convex_polygon import calculate_area, calculate_intersection_area
from scripts.environments.pusht.observation import (
    ARENA_SIZE,
    BLOCK_ORIENTATION_INDEX,
    BLOCK_POSITION_SLICE,
    POSITION_SLICE,
)


def calculate_reward(
    *,
    state_batch: npt.NDArray[np.float32],  # (batch_size, state_dimensionality)
) -> npt.NDArray[np.floating]:  # (batch_size,)
    environment = _create_environment()
    return np.clip(
        a=_calculate_coverage(state_batch=state_batch, environment=environment)
        / environment.success_threshold,
        a_min=0.0,
        a_max=1.0,
    )


def calculate_termination(
    *,
    state_batch: npt.NDArray[np.float32],  # (batch_size, state_dimensionality)
) -> npt.NDArray[np.bool_]:  # (batch_size,)
    position_batch = state_batch[:, POSITION_SLICE]  # (batch_size, 4)
    reachable_batch: npt.NDArray[np.bool_] = np.asarray(
        a=np.isfinite(state_batch).all(axis=-1)
        & (position_batch > -ARENA_SIZE).all(axis=-1)
        & (position_batch < 2 * ARENA_SIZE).all(axis=-1)
    )  # (batch_size,)

    termination_batch: npt.NDArray[np.bool_] = ~reachable_batch  # (batch_size,)
    if not reachable_batch.any():
        return termination_batch

    environment = _create_environment()
    termination_batch[reachable_batch] = np.asarray(
        a=_calculate_coverage(
            state_batch=state_batch[reachable_batch], environment=environment
        )
        > environment.success_threshold
    )
    return termination_batch


# ---- Private --------------------------------------------


def _create_environment() -> PushTEnv:
    environment = PushTEnv()
    environment.reset()
    return environment


def _calculate_coverage(
    *,
    state_batch: npt.NDArray[np.float32],  # (batch_size, state_dimensionality)
    environment: PushTEnv,
) -> npt.NDArray[np.floating]:  # (batch_size,)
    block_shapes = list(environment.block.shapes)

    goal_body = PushTEnv.get_goal_pose_body(environment.goal_pose)
    goal_corner_batch = np.stack(
        arrays=[
            [list(goal_body.local_to_world(corner)) for corner in shape.get_vertices()]
            for shape in block_shapes
        ]
    )  # (part_count, corner_count, 2)

    block_corner_batch = _place_block(
        state_batch=state_batch.astype(np.float64), block_shapes=block_shapes
    )  # (batch_size, part_count, corner_count, 2)

    # The T is made of two parts, so every part of the block is laid over every
    # part of the goal. The parts only touch, so their overlaps add up.
    part_count = len(block_shapes)
    block_part_indices = np.repeat(a=np.arange(part_count), repeats=part_count)
    goal_part_indices = np.tile(A=np.arange(part_count), reps=part_count)

    pair_count = part_count * part_count
    batch_size = len(state_batch)

    overlap_area_batch = calculate_intersection_area(
        first_corner_batch=block_corner_batch[:, block_part_indices].reshape(
            batch_size * pair_count, -1, 2
        ),
        second_corner_batch=np.broadcast_to(
            goal_corner_batch[goal_part_indices],
            (batch_size, *goal_corner_batch[goal_part_indices].shape),
        ).reshape(batch_size * pair_count, -1, 2),
    )  # (batch_size * pair_count,)

    return overlap_area_batch.reshape(batch_size, pair_count).sum(axis=1) / (
        calculate_area(corner_batch=goal_corner_batch).sum()
    )


def _place_block(
    *,
    state_batch: np.ndarray,  # (batch_size, state_dimensionality)
    block_shapes: list,
) -> np.ndarray:  # (batch_size, part_count, corner_count, 2)
    local_corner_batch = np.stack(
        arrays=[
            [list(corner) for corner in shape.get_vertices()] for shape in block_shapes
        ]
    )  # (part_count, corner_count, 2)

    angle_batch = state_batch[:, BLOCK_ORIENTATION_INDEX]  # (batch_size,)
    cosine_batch = np.cos(angle_batch)
    sine_batch = np.sin(angle_batch)
    rotation_batch = np.stack(
        arrays=[
            np.stack(arrays=[cosine_batch, -sine_batch], axis=-1),
            np.stack(arrays=[sine_batch, cosine_batch], axis=-1),
        ],
        axis=-2,
    )  # (batch_size, 2, 2)

    return (
        np.einsum("bij,pcj->bpci", rotation_batch, local_corner_batch)
        + state_batch[:, None, None, BLOCK_POSITION_SLICE]
    )
