"""The one exact symmetry PushT offers.

A rotation or a translation of a whole episode would be exact only if the goal
moved with it. The goal is fixed at (256, 256, pi/4) and does not appear in the
observation, so it cannot move. What remains is the subgroup that maps the goal
onto itself.

That subgroup is not empty. The T block is symmetric about its own stem, so a
reflection about the stem axis of the goal pose maps the goal T onto itself.
The arena is a square whose diagonals run at that same angle, so the walls map
onto themselves too.

Two facts make this exact rather than approximate:

  * every state and action is a position or an orientation, and both transform
    in closed form -- no model, no tolerance, no error to account for;
  * the reflected episode solves the same task against the same goal, so the
    recorded actions stay correct labels.

One pixel of error remains. The walls run from (5, 5) to (506, 506) and are
therefore centred on (255.5, 255.5), while the goal sits at (256, 256). The
reflection is taken about the goal, so the walls land one pixel off their true
image. That is far below the block scale of 30 and the agent radius of 15.

The physics is assumed to be reflection invariant. Pymunk's friction and
collision response carry no handedness, so a mirrored world evolves into the
mirror of what the original world evolves into.
"""

import numpy as np
from numpy import typing as npt

from scripts.shared.pusht_state import (
    BLOCK_ORIENTATION_INDEX,
    BLOCK_POSITION_SLICE,
    FULL_TURN,
    AGENT_POSITION_SLICE,
)

# The pose the block is pushed towards, as hard-coded by the environment.
GOAL_POSITION: tuple[float, float] = (256.0, 256.0)
GOAL_ORIENTATION = np.pi / 4.0

# The block's own axis of symmetry is its stem, which in the body frame points
# along +y. At orientation theta it points at theta + pi/2 in the world, so
# this is the line the goal T is mirrored about.
REFLECTION_AXIS_ANGLE = GOAL_ORIENTATION + np.pi / 2.0


def reflect_positions(
    *,
    position_batch: npt.NDArray[np.float32],  # (batch_size, 2)
) -> npt.NDArray[np.float32]:  # (batch_size, 2)
    center = np.asarray(a=GOAL_POSITION, dtype=np.float64)
    offset_batch = position_batch.astype(np.float64) - center
    return (center + offset_batch @ _build_reflection_matrix().T).astype(np.float32)


def reflect_orientations(
    *,
    orientation_batch: npt.NDArray[np.float32],  # (batch_size,)
) -> npt.NDArray[np.float32]:  # (batch_size,)
    """Mirror an orientation, using the block's own symmetry.

    A reflection reverses handedness, so the mirrored block is not a rotation
    of the original one. It is a rotation of the original one composed with the
    block's own mirror symmetry, which leaves the shape unchanged -- and that
    composition is again a plain orientation.
    """
    return np.asarray(
        a=(2.0 * REFLECTION_AXIS_ANGLE - np.pi - orientation_batch.astype(np.float64))
        % FULL_TURN,
        dtype=np.float32,
    )


def reflect_states(
    *,
    state_batch: npt.NDArray[np.float32],  # (batch_size, state_dimensionality)
) -> npt.NDArray[np.float32]:  # (batch_size, state_dimensionality)
    reflected_batch = state_batch.copy()
    reflected_batch[:, AGENT_POSITION_SLICE] = reflect_positions(
        position_batch=state_batch[:, AGENT_POSITION_SLICE]
    )
    reflected_batch[:, BLOCK_POSITION_SLICE] = reflect_positions(
        position_batch=state_batch[:, BLOCK_POSITION_SLICE]
    )
    reflected_batch[:, BLOCK_ORIENTATION_INDEX] = reflect_orientations(
        orientation_batch=state_batch[:, BLOCK_ORIENTATION_INDEX]
    )
    return reflected_batch


def reflect_actions(
    *,
    action_batch: npt.NDArray[np.float32],  # (batch_size, action_dimensionality)
) -> npt.NDArray[np.float32]:  # (batch_size, action_dimensionality)
    """An action is the position the agent is sent to, so it mirrors like one."""
    return reflect_positions(position_batch=action_batch)


def measure_goal_pose_drift() -> float:
    """How far the reflection moves the goal pose it is built around.

    The transform is only a valid relabelling while the goal is a fixed point.
    This is that claim as a number, so a run can assert it rather than trust
    the derivation above.
    """
    reflected_position = reflect_positions(
        position_batch=np.asarray(a=[GOAL_POSITION], dtype=np.float32)
    )[0]
    reflected_orientation = reflect_orientations(
        orientation_batch=np.asarray(a=[GOAL_ORIENTATION], dtype=np.float32)
    )[0]

    position_drift = float(
        np.linalg.norm(reflected_position - np.asarray(a=GOAL_POSITION))
    )
    orientation_drift = float(
        np.abs((reflected_orientation - GOAL_ORIENTATION + np.pi) % FULL_TURN - np.pi)
    )
    return max(position_drift, orientation_drift)


# ---- Private --------------------------------------------


def _build_reflection_matrix() -> npt.NDArray[np.float64]:  # (2, 2)
    double_angle = 2.0 * REFLECTION_AXIS_ANGLE
    return np.asarray(
        a=[
            [np.cos(double_angle), np.sin(double_angle)],
            [np.sin(double_angle), -np.cos(double_angle)],
        ],
        dtype=np.float64,
    )
