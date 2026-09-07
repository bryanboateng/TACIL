import numpy as np
import pymunk
from gym_pusht.envs.pusht import PushTEnv, pymunk_to_shapely

from scripts.shared.pusht_state import BLOCK_ORIENTATION_INDEX, BLOCK_POSITION_SLICE

GOAL_POSE = np.array([256.0, 256.0, np.pi / 4.0])

SUCCESS_THRESHOLD = 0.95


def calculate_coverage_batch(
    *,
    state_batch: np.ndarray,  # (batch_size, state_dimensionality)
) -> np.ndarray:  # (batch_size,)
    block_body, block_shapes = _create_block()

    goal_geometry = pymunk_to_shapely(
        PushTEnv.get_goal_pose_body(GOAL_POSE), block_shapes
    )
    goal_area = goal_geometry.area

    coverage_batch = np.empty(shape=len(state_batch))
    for index, state in enumerate(state_batch):
        # Set the angle first. A body turns around its center of gravity, so
        # setting the angle after the position moves the body away again.
        block_body.angle = float(state[BLOCK_ORIENTATION_INDEX])
        block_body.position = state[BLOCK_POSITION_SLICE].tolist()

        block_geometry = pymunk_to_shapely(block_body, block_shapes)
        coverage_batch[index] = (
            block_geometry.intersection(goal_geometry).area / goal_area
        )

    return coverage_batch


def calculate_reward_batch(
    *,
    state_batch: np.ndarray,  # (batch_size, state_dimensionality)
) -> np.ndarray:  # (batch_size,)
    return np.clip(
        calculate_coverage_batch(state_batch=state_batch) / SUCCESS_THRESHOLD, 0.0, 1.0
    )


def calculate_termination_batch(
    *,
    state_batch: np.ndarray,  # (batch_size, state_dimensionality)
) -> np.ndarray:  # (batch_size,)
    return calculate_coverage_batch(state_batch=state_batch) > SUCCESS_THRESHOLD


# ---- Private --------------------------------------------


def _create_block() -> tuple[pymunk.Body, list[pymunk.Shape]]:
    # The shapes carry the T outline. The throwaway space is only needed
    # because the environment builds the block by adding it to a space.
    return PushTEnv.add_tee(space=pymunk.Space(), position=(0.0, 0.0), angle=0.0)
