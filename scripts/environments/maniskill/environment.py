import gymnasium
import mani_skill.envs  # noqa: F401
import numpy as np
import sapien
from gymnasium import Env
from gymnasium.spaces import Box
from mani_skill.utils.wrappers import CPUGymWrapper

from scripts.environments.conditions import Condition, DisturbanceLevel, FloorCondition
from scripts.environments.maniskill.disturbance import DisturbanceWrapper
from scripts.environments.maniskill.gpi_geometry import ManiSkillGpiGeometry
from scripts.environments.maniskill.observation import (
    POSE_ACTION_POSITION_SLICE,
    control_mode,
    create_state_space,
    has_absolute_pose_actions,
    kept_column_indices,
    known_task_ids,
    max_episode_steps,
    robot_root_translation,
)
from scripts.environments.vector_state_space import VectorStateSpace
from scripts.evaluation.core.rollout import EpisodeRecord

# A sparse reward is one for a solved episode and zero otherwise, so the mean
# reward is the success rate. That is the number ManiSkill reports for its own
# baselines, which keeps the results comparable.
_REWARD_MODE = "sparse"

# The renderer hands its pictures to torch, and the GPU path needs a card new
# enough for the installed torch build. The processor path always works and
# only ever draws the handful of episodes that go into a video.
_RENDER_BACKEND = "cpu"

# How quickly the moved object loses speed on the table. GRIPPY is the setting
# the demonstrations were recorded under, SLIPPERY the altered one.
#
# For a rolling ball this is damping, not surface friction. Friction decides
# how a body slides, and a ball hardly slides: raising it from 0.05 to 0.30
# changed the roll distance by three percent, which is no condition at all.
# Damping decides how a body coasts, and that is what a ball does.
#
# Because the recorded setting already lets the ball coast almost freely, the
# only direction with any room left is more resistance. So here the altered
# surface is a slower one. Measured with a 0.6 m/s push: 0.00 rolls 1.96 m,
# 0.25 rolls 1.54 m, 0.50 rolls 1.33 m, 1.00 rolls 0.42 m. The goal sits
# roughly half a metre away, so 0.50 shortens the roll clearly while leaving
# the task solvable.
_DAMPING_BY_FLOOR_CONDITION = {FloorCondition.GRIPPY: 0.0, FloorCondition.SLIPPERY: 0.5}

# The strength of one sideways kick, in newton seconds. Measured as the
# distance the ball ends up from where it would have been: 0.002 moves it
# 0.06 m, 0.005 moves it 0.16 m. Both are a visible fraction of the half metre
# to the goal without making the task hopeless.
_IMPULSE_BY_DISTURBANCE_LEVEL = {
    DisturbanceLevel.LOW: 0.002,
    DisturbanceLevel.HIGH: 0.005,
}

# The name the task gives the object the policy moves.
_MOVED_OBJECT_NAME_BY_TASK = {"RollBall-v1": "ball", "PushCube-v1": "obj"}


class ManiSkillEnvironment:
    """A Panda arm on a table, driven through ManiSkill."""

    def __init__(self, *, task_id: str) -> None:
        if task_id not in _MOVED_OBJECT_NAME_BY_TASK:
            raise KeyError(
                f"Unknown ManiSkill task {task_id!r}. Known tasks: {known_task_ids()}"
            )
        self._task_id = task_id
        self._state_space = create_state_space(task_id=task_id)

    @property
    def state_space(self) -> VectorStateSpace:
        return self._state_space

    @property
    def gpi_geometry(self) -> ManiSkillGpiGeometry:
        return ManiSkillGpiGeometry(task_id=self._task_id)

    @property
    def video_frames_per_second(self) -> int:
        return 20

    def create_gym_environment(self, *, condition: Condition, render: bool) -> Env:
        moved_object_name = _MOVED_OBJECT_NAME_BY_TASK[self._task_id]

        environment: Env = CPUGymWrapper(
            gymnasium.make(
                self._task_id,
                obs_mode="state",
                control_mode=control_mode(task_id=self._task_id),
                reward_mode=_REWARD_MODE,
                sim_backend="physx_cpu",
                render_backend=_RENDER_BACKEND,
                render_mode="rgb_array" if render else None,
                max_episode_steps=max_episode_steps(task_id=self._task_id),
            )
        )

        if has_absolute_pose_actions(task_id=self._task_id):
            environment = _RootFrameActionWrapper(
                environment=environment,
                root_translation=robot_root_translation(task_id=self._task_id),
            )

        environment = _SelectedObservationWrapper(
            environment=environment,
            column_indices=kept_column_indices(task_id=self._task_id),
        )

        environment = _DampingWrapper(
            environment=environment,
            object_name=moved_object_name,
            damping=_DAMPING_BY_FLOOR_CONDITION[condition.floor_condition],
        )

        if condition.disturbance_level is not None:
            environment = DisturbanceWrapper(
                environment=environment,
                object_name=moved_object_name,
                magnitude=_IMPULSE_BY_DISTURBANCE_LEVEL[condition.disturbance_level],
            )

        return environment

    def render_episode_frames(
        self,
        *,
        episode_record: EpisodeRecord,
        condition: Condition,
        policy_label: str,
        percentile: int,
    ) -> list[np.ndarray]:
        if episode_record.frames is None:
            raise ValueError(f"No frames found for episode {episode_record.seed}")
        return episode_record.frames


# ---- Private --------------------------------------------


class _RootFrameActionWrapper(gymnasium.ActionWrapper):
    """Takes a gripper position measured from the world origin and hands the
    simulator the same position measured from the robot's base.

    The pipeline works in world coordinates throughout, because that is the
    frame the state reports the gripper in, and GPI subtracts one from the
    other.
    """

    def __init__(
        self, *, environment: gymnasium.Env, root_translation: np.ndarray
    ) -> None:
        super().__init__(env=environment)
        self._root_translation = root_translation

    def action(self, action: np.ndarray) -> np.ndarray:
        root_frame_action = np.asarray(a=action, dtype=np.float32).copy()
        root_frame_action[POSE_ACTION_POSITION_SLICE] -= self._root_translation
        return root_frame_action


class _SelectedObservationWrapper(gymnasium.ObservationWrapper):
    """Keeps only the state entries the pipeline works on.

    See `observation.py` for which entries are dropped and why.
    """

    def __init__(
        self, *, environment: gymnasium.Env, column_indices: np.ndarray
    ) -> None:
        super().__init__(env=environment)
        self._column_indices = column_indices
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(len(column_indices),), dtype=np.float32
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return observation[self._column_indices]


class _DampingWrapper(gymnasium.Wrapper):
    """Sets how quickly the moved object loses speed on the table.

    The setting is applied after every reset, because a reset can rebuild the
    scene and hand back a fresh body.
    """

    # The recorded setting damps the spin a little even where it damps the
    # travel not at all, so the spin never drops below that.
    _MINIMUM_ANGULAR_DAMPING = 0.05

    def __init__(
        self, *, environment: gymnasium.Env, object_name: str, damping: float
    ) -> None:
        super().__init__(env=environment)
        self._object_name = object_name
        self._damping = damping

    def reset(self, **kwargs) -> tuple:
        result = self.env.reset(**kwargs)
        self._apply_damping()
        return result

    def _apply_damping(self) -> None:
        actor = getattr(self.env.unwrapped, self._object_name)
        body = actor._objs[0].find_component_by_type(
            sapien.physx.PhysxRigidDynamicComponent
        )
        body.linear_damping = self._damping
        body.angular_damping = max(self._MINIMUM_ANGULAR_DAMPING, self._damping)
