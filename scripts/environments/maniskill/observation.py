import numpy as np
from numpy import typing as npt

from scripts.environments.state_space import PositionGroup, StatePicture
from scripts.environments.vector_state_space import VectorStateSpace

# ManiSkill flattens its state dictionary in a fixed order, so every entry sits
# at a known offset. The offsets below were read off the `state_dict`
# observation of each task.
#
# PushCube-v1, 35 entries:
#   qpos 0:9, qvel 9:18, tcp_pose 18:25, goal_pos 25:28, obj_pose 28:35
_FULL_DIMENSIONALITY_BY_TASK: dict[str, int] = {"PushCube-v1": 35}

# A pose ends in a quaternion, which is a rotation written as four numbers.
# Both quaternions are dropped, because neither carries anything.
#
# The gripper is held in one orientation for the whole push: measured over all
# 998 recordings it turns by 0.00 degrees per step, and by 0.27 degrees even at
# the 99th percentile. The cube, lying flat and pushed across a table, turns by
# 0.00 degrees per step and 1.55 at the 99th percentile. Neither task needs
# them either: the gripper orientation follows from the joint angles the state
# already carries, and the cube's target is a position.
#
# Dropping them also keeps a spread problem out of the state. Every entry is
# divided by its own spread before the dynamics model and the uncertainty gate
# see it, so an entry that barely moves is amplified the most. The quaternion
# entries spread by as little as 1.7e-4, against 5.4e-3 for the tightest joint
# angle.
#
# What is left are positions, speeds and joint angles, which all add and
# subtract like ordinary numbers.
_DROPPED_SLICES_BY_TASK: dict[str, tuple[slice, ...]] = {
    # tcp_pose quaternion, obj_pose quaternion
    "PushCube-v1": (slice(21, 25), slice(31, 35))
}

# Where the gripper's own position sits in a raw state. It is the position
# half of `tcp_pose`.
_RAW_TCP_POSITION_SLICE = slice(18, 21)

# Where the pushed object's own position sits in a raw state. It is the
# position half of the object's pose.
_RAW_OBJECT_POSITION_SLICE_BY_TASK: dict[str, slice] = {"PushCube-v1": slice(28, 31)}

# The part of the table a diagnostic plot has to show, in metres, as
# (horizontal limits, vertical limits).
#
# PushCube drops the cube anywhere in x and y between -0.1 and 0.1, and puts
# the goal 0.2 m further along x. The two of them therefore span x from -0.1
# to 0.3. The margin leaves room for the gripper, which reaches around the
# cube to push it.
_PICTURE_LIMITS_BY_TASK: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "PushCube-v1": ((-0.2, 0.35), (-0.15, 0.15))
}

# How each task is driven.
#
# `pd_ee_pose` asks for an absolute gripper pose plus a gripper opening, as
# three positions, three rotation entries and one opening. GPI needs that
# absolute form, because it subtracts a recorded position from a recorded
# action and that only means something when the action is a position.
#
# The recordings are joint angles from a motion planner, which ManiSkill can
# convert into this form. It cannot convert the other way round: a recording of
# gripper displacements can never become an absolute pose.
_CONTROL_MODE_BY_TASK: dict[str, str] = {"PushCube-v1": "pd_ee_pose"}

_ACTION_DIMENSIONALITY_BY_CONTROL_MODE: dict[str, int] = {"pd_ee_pose": 7}

# How long an episode may run.
#
# ManiSkill's own limit for PushCube is 50 steps, but the motion planner that
# recorded its demonstrations needs 61 to 148, mean 69. Scored under the
# registered limit not even the expert solves the task. ManiSkill's own
# diffusion policy baseline raises it to 100, which is the number used here.
_MAX_EPISODE_STEPS_BY_TASK: dict[str, int] = {"PushCube-v1": 100}

# Where the robot stands, read off `agent.robot.get_pose().p`.
#
# A `pd_ee_pose` action names the wanted gripper position measured from the
# robot's own base, while the state reports the gripper position measured from
# the world origin. The two therefore differ by exactly this translation. The
# offset has to go before anything is normalized, because states and actions
# are normalized separately and a shift does not cancel out afterwards.
_ROBOT_ROOT_TRANSLATION_BY_TASK: dict[str, npt.NDArray[np.float32]] = {
    "PushCube-v1": np.array(object=[-0.615, 0.0, 0.0], dtype=np.float32)
}

# The entries of a `pd_ee_pose` action that name a position. The rest is
# rotation and gripper opening.
POSE_ACTION_POSITION_SLICE = slice(0, 3)


def kept_column_indices(*, task_id: str) -> npt.NDArray[np.int64]:  # (kept_count,)
    """The entries of a raw ManiSkill state that the pipeline works on."""
    _check_task(task_id=task_id)

    keep_mask = np.ones(shape=_FULL_DIMENSIONALITY_BY_TASK[task_id], dtype=bool)
    for dropped_slice in _DROPPED_SLICES_BY_TASK[task_id]:
        keep_mask[dropped_slice] = False

    return np.flatnonzero(keep_mask)


def agent_position_slice(*, task_id: str) -> slice:
    """Where the gripper position sits once the dropped entries are gone."""
    return _kept_slice(
        task_id=task_id, raw_slice=_RAW_TCP_POSITION_SLICE, label="gripper position"
    )


def object_position_slice(*, task_id: str) -> slice:
    """Where the pushed object's position sits once the dropped entries are
    gone."""
    _check_task(task_id=task_id)
    return _kept_slice(
        task_id=task_id,
        raw_slice=_RAW_OBJECT_POSITION_SLICE_BY_TASK[task_id],
        label="object position",
    )


def max_episode_steps(*, task_id: str) -> int:
    _check_task(task_id=task_id)
    return _MAX_EPISODE_STEPS_BY_TASK[task_id]


def control_mode(*, task_id: str) -> str:
    _check_task(task_id=task_id)
    return _CONTROL_MODE_BY_TASK[task_id]


def action_dimensionality(*, task_id: str) -> int:
    return _ACTION_DIMENSIONALITY_BY_CONTROL_MODE[control_mode(task_id=task_id)]


def robot_root_translation(*, task_id: str) -> npt.NDArray[np.float32]:  # (3,)
    _check_task(task_id=task_id)
    return _ROBOT_ROOT_TRANSLATION_BY_TASK[task_id]


def create_state_space(*, task_id: str) -> VectorStateSpace:
    horizontal_limits, vertical_limits = _PICTURE_LIMITS_BY_TASK[task_id]
    return VectorStateSpace(
        dimensionality=len(kept_column_indices(task_id=task_id)),
        action_dimensionality=action_dimensionality(task_id=task_id),
        picture=StatePicture(
            position_groups=(
                PositionGroup(
                    label="Gripper Position",
                    entries=_horizontal_plane(
                        entries=agent_position_slice(task_id=task_id)
                    ),
                ),
                PositionGroup(
                    label="Object Position",
                    entries=_horizontal_plane(
                        entries=object_position_slice(task_id=task_id)
                    ),
                ),
            ),
            horizontal_limits=horizontal_limits,
            vertical_limits=vertical_limits,
            position_unit="m",
            # The quaternions were dropped, so no angle is left to draw.
            orientation_index=None,
        ),
    )


def known_task_ids() -> list[str]:
    return sorted(_FULL_DIMENSIONALITY_BY_TASK)


# ---- Private --------------------------------------------


def _horizontal_plane(*, entries: slice) -> slice:
    """Drop the height from a position, because a plot is flat."""
    return slice(entries.start, entries.start + 2)


def _kept_slice(*, task_id: str, raw_slice: slice, label: str) -> slice:
    kept = kept_column_indices(task_id=task_id)
    positions = np.flatnonzero(
        np.isin(element=kept, test_elements=np.arange(raw_slice.start, raw_slice.stop))
    )
    if len(positions) != 3 or not np.array_equal(np.diff(positions), [1, 1]):
        raise ValueError(
            f"The {label} of {task_id!r} is no longer three entries in a row"
        )
    return slice(int(positions[0]), int(positions[-1]) + 1)


def _check_task(*, task_id: str) -> None:
    if task_id not in _FULL_DIMENSIONALITY_BY_TASK:
        raise KeyError(
            f"Unknown ManiSkill task {task_id!r}. Known tasks: {known_task_ids()}"
        )
