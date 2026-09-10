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
    kept_column_indices,
    known_task_ids,
    max_episode_steps,
    robot_root_translation,
)
from scripts.environments.vector_state_space import VectorStateSpace
from scripts.evaluation.core import video
from scripts.evaluation.core.rollout import EpisodeRecord

# A sparse reward is one for a solved episode and zero otherwise, so the mean
# reward is the success rate. That is the number ManiSkill reports for its own
# baselines, which keeps the results comparable.
_REWARD_MODE = "sparse"

# The renderer hands its pictures to torch, and the GPU path needs a card new
# enough for the installed torch build. The processor path always works and
# only ever draws the handful of episodes that go into a video.
_RENDER_BACKEND = "cpu"

# How well the pushed object grips the table. GRIPPY is the setting the
# demonstrations were recorded under, SLIPPERY the altered one.
#
# For a cube this is surface friction, because a cube slides. Measured as how
# far the cube travels after a 0.5 m/s knock: friction 0.30 lets it slide
# 0.04 m, 0.10 lets it slide 0.10 m and 0.05 lets it slide 0.11 m. The chosen
# pair therefore roughly triples the slide without making the cube
# uncontrollable.
_FRICTION_BY_FLOOR_CONDITION = {
    FloorCondition.GRIPPY: 0.3,
    FloorCondition.SLIPPERY: 0.05,
}

# The strength of one sideways kick, in newton seconds. Measured as how far the
# cube ends up from where it would have been after four kicks: 0.010 moves it
# 0.014 m and 0.020 moves it 0.060 m. Both are a visible fraction of the
# roughly 0.1 m to the goal without knocking the cube off the table, which
# 0.050 does at 0.40 m.
_IMPULSE_BY_DISTURBANCE_LEVEL = {
    DisturbanceLevel.LOW: 0.01,
    DisturbanceLevel.HIGH: 0.02,
}

# The name the task gives the object the policy moves.
_MOVED_OBJECT_NAME_BY_TASK = {"PushCube-v1": "obj"}


class ManiSkillEnvironment:
    """A Panda arm on a table, driven through ManiSkill."""

    def __init__(self, *, task_id: str) -> None:
        if task_id not in _MOVED_OBJECT_NAME_BY_TASK:
            raise KeyError(
                f"Unknown ManiSkill task {task_id!r}. Known tasks: {known_task_ids()}"
            )
        self._task_id = task_id
        self._state_space = create_state_space(task_id=task_id)
        self._gpi_geometry = ManiSkillGpiGeometry(task_id=task_id)

    @property
    def state_space(self) -> VectorStateSpace:
        return self._state_space

    @property
    def gpi_geometry(self) -> ManiSkillGpiGeometry:
        return self._gpi_geometry

    @property
    def video_frames_per_second(self) -> int:
        return 20

    def create_gym_environment(self, *, condition: Condition, render: bool) -> Env:
        moved_object_name = _MOVED_OBJECT_NAME_BY_TASK[self._task_id]

        environment: Env = _RebuiltSceneWrapper(
            environment=CPUGymWrapper(
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
        )

        environment = _RootFrameActionWrapper(
            environment=environment,
            root_translation=robot_root_translation(task_id=self._task_id),
        )

        environment = _SelectedObservationWrapper(
            environment=environment,
            column_indices=kept_column_indices(task_id=self._task_id),
        )

        environment = _FrictionWrapper(
            environment=environment,
            object_name=moved_object_name,
            friction=_FRICTION_BY_FLOOR_CONDITION[condition.floor_condition],
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
        return video.render_episode_frames(
            episode_record=episode_record,
            condition=condition,
            policy_label=policy_label,
            percentile=percentile,
            # ManiSkill records 20 frames a second, which is watchable as it is.
            playback_speed=1.0,
        )


# ---- Private --------------------------------------------


class _RebuiltSceneWrapper(gymnasium.Wrapper):
    """Builds the scene again on every reset.

    PhysX keeps solver state between episodes, such as its record of which
    surfaces touch. A plain reset leaves that state in place, so an episode
    played a second time in the same environment drifts away from the first.
    The evaluation plays a scored episode again to record it, and refuses a
    recording that does not match, so the two have to agree exactly.
    """

    def __init__(self, *, environment: gymnasium.Env) -> None:
        super().__init__(env=environment)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple:
        return self.env.reset(
            seed=seed, options={**(options or {}), "reconfigure": True}
        )


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


class _FrictionWrapper(gymnasium.Wrapper):
    """Sets how well the pushed object grips the table.

    The setting is applied after every reset, because a reset can rebuild the
    scene and hand back a fresh material.
    """

    def __init__(
        self, *, environment: gymnasium.Env, object_name: str, friction: float
    ) -> None:
        super().__init__(env=environment)
        self._object_name = object_name
        self._friction = friction

    def reset(self, **kwargs) -> tuple:
        result = self.env.reset(**kwargs)
        self._apply_friction()
        return result

    def _apply_friction(self) -> None:
        actor = getattr(self.env.unwrapped, self._object_name)
        body = actor._objs[0].find_component_by_type(
            sapien.physx.PhysxRigidDynamicComponent
        )
        for collision_shape in body.collision_shapes:
            collision_shape.physical_material.static_friction = self._friction
            collision_shape.physical_material.dynamic_friction = self._friction
