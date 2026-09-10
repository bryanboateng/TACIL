import numpy as np
from numpy import typing as npt

from scripts.environments.vector_state_space import VectorStateSpace

# ManiSkill flattens its state dictionary in a fixed order, so every entry sits
# at a known offset. The offsets below were read off the `state_dict`
# observation of each task.
#
# RollBall-v1, 44 entries:
#   qpos 0:9, qvel 9:18, tcp_pose 18:25, goal_pos 25:28, ball_pose 28:35,
#   ball_vel 35:38, tcp_to_ball_pos 38:41, ball_to_goal_pos 41:44
#
# PushCube-v1, 35 entries:
#   qpos 0:9, qvel 9:18, tcp_pose 18:25, goal_pos 25:28, obj_pose 28:35
_FULL_DIMENSIONALITY_BY_TASK: dict[str, int] = {"RollBall-v1": 44, "PushCube-v1": 35}

# A pose ends in a quaternion, which is a rotation written as four numbers.
# Those four numbers are dropped, for two reasons measured on the recorded
# data.
#
# The gripper quaternion follows from the joint angles, which the state
# already carries, so it adds nothing.
#
# The object quaternion cannot be predicted. Between two recorded steps the
# ball turns by 80 degrees on average and by up to 180, so its orientation is
# sampled far too coarsely to carry information. A dynamics model would burn
# capacity on noise, and the replay error that gates the augmentation would be
# swamped by it. A rolling ball does not need an orientation either: a sphere
# looks the same whichever way it lies. The same holds for the flat lying cube
# in PushCube, whose target is a position only.
#
# What is left are positions, speeds and joint angles, which all add and
# subtract like ordinary numbers.
_DROPPED_SLICES_BY_TASK: dict[str, tuple[slice, ...]] = {
    # tcp_pose quaternion, ball_pose quaternion
    "RollBall-v1": (slice(21, 25), slice(31, 35)),
    # tcp_pose quaternion, obj_pose quaternion
    "PushCube-v1": (slice(21, 25), slice(31, 35)),
}

# Where the gripper's own position sits in a raw state. It is the position
# half of `tcp_pose`.
_RAW_TCP_POSITION_SLICE = slice(18, 21)

# How each task is driven.
#
# `pd_ee_delta_pos` asks for a gripper displacement plus a gripper opening.
# `pd_ee_pose` asks for an absolute gripper pose plus a gripper opening, as
# three positions, three rotation entries and one opening.
#
# GPI needs the absolute mode, because it subtracts a recorded position from a
# recorded action and that only means something when the action is a position.
# RollBall is stuck on the relative mode: its recordings are gripper
# displacements collected on the graphics card, and ManiSkill refuses to turn
# those into absolute poses ("Script currently does not support converting
# pd_ee_delta_pos to pd_ee_pose"). PushCube was solved by a motion planner in
# joint angles, which does convert.
_CONTROL_MODE_BY_TASK: dict[str, str] = {
    "RollBall-v1": "pd_ee_delta_pos",
    "PushCube-v1": "pd_ee_pose",
}

_ACTION_DIMENSIONALITY_BY_CONTROL_MODE: dict[str, int] = {
    "pd_ee_delta_pos": 4,
    "pd_ee_pose": 7,
}

# How long an episode may run.
#
# ManiSkill's own limit for PushCube is 50 steps, but the motion planner that
# recorded its demonstrations needs 61 to 148, mean 69. Scored under the
# registered limit not even the expert solves the task. ManiSkill's own
# diffusion policy baseline raises it to 100, which is the number used here.
#
# RollBall keeps its registered 80, which already covers its longest
# recording of 68 steps.
_MAX_EPISODE_STEPS_BY_TASK: dict[str, int] = {"RollBall-v1": 80, "PushCube-v1": 100}


# Where the robot stands, read off `agent.robot.get_pose().p`.
#
# A `pd_ee_pose` action names the wanted gripper position measured from the
# robot's own base, while the state reports the gripper position measured from
# the world origin. The two therefore differ by exactly this translation. The
# offset has to go before anything is normalized, because states and actions
# are normalized separately and a shift does not cancel out afterwards.
_ROBOT_ROOT_TRANSLATION_BY_TASK: dict[str, npt.NDArray[np.float32]] = {
    "RollBall-v1": np.array(object=[-0.1, 1.0, 0.0], dtype=np.float32),
    "PushCube-v1": np.array(object=[-0.615, 0.0, 0.0], dtype=np.float32),
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
    kept = kept_column_indices(task_id=task_id)
    positions = np.flatnonzero(
        np.isin(
            element=kept,
            test_elements=np.arange(
                _RAW_TCP_POSITION_SLICE.start, _RAW_TCP_POSITION_SLICE.stop
            ),
        )
    )
    if len(positions) != 3 or not np.array_equal(np.diff(positions), [1, 1]):
        raise ValueError(
            f"The gripper position of {task_id!r} is no longer three entries in a row"
        )
    return slice(int(positions[0]), int(positions[-1]) + 1)


def max_episode_steps(*, task_id: str) -> int:
    _check_task(task_id=task_id)
    return _MAX_EPISODE_STEPS_BY_TASK[task_id]


def control_mode(*, task_id: str) -> str:
    _check_task(task_id=task_id)
    return _CONTROL_MODE_BY_TASK[task_id]


def action_dimensionality(*, task_id: str) -> int:
    return _ACTION_DIMENSIONALITY_BY_CONTROL_MODE[control_mode(task_id=task_id)]


def has_absolute_pose_actions(*, task_id: str) -> bool:
    """Whether an action names a gripper position, which is what GPI needs."""
    return control_mode(task_id=task_id) == "pd_ee_pose"


def robot_root_translation(*, task_id: str) -> npt.NDArray[np.float32]:  # (3,)
    _check_task(task_id=task_id)
    return _ROBOT_ROOT_TRANSLATION_BY_TASK[task_id]


def create_state_space(*, task_id: str) -> VectorStateSpace:
    return VectorStateSpace(
        dimensionality=len(kept_column_indices(task_id=task_id)),
        action_dimensionality=action_dimensionality(task_id=task_id),
    )


def known_task_ids() -> list[str]:
    return sorted(_FULL_DIMENSIONALITY_BY_TASK)


# ---- Private --------------------------------------------


def _check_task(*, task_id: str) -> None:
    if task_id not in _FULL_DIMENSIONALITY_BY_TASK:
        raise KeyError(
            f"Unknown ManiSkill task {task_id!r}. Known tasks: {known_task_ids()}"
        )
