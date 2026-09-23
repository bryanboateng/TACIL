from typing import Literal

import gymnasium
import mani_skill.envs  # noqa: F401
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box
from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv
from mani_skill.utils.wrappers import CPUGymWrapper
from numpy import typing as npt

from scripts.environments import state_bounds
from scripts.environments.condition import Condition
from scripts.environments.peg_insertion import reward
from scripts.environments.peg_insertion.condition import (
    ALL_CONDITIONS,
    SETTINGS_BY_CONDITION,
)
from scripts.environments.peg_insertion.disturbance import DisturbanceWrapper
from scripts.environments.peg_insertion.gpi_geometry import PegInsertionGpiGeometry
from scripts.environments.peg_insertion.observation import (
    HOLE_POSITION_SLICE,
    HOLE_RADIUS_INDEX,
    PEG_HALF_SIZE_SLICE,
    PEG_POSITION_SLICE,
    STATE_DIMENSIONALITY,
    TCP_POSITION_SLICE,
    extract_state_from_environment,
)
from scripts.environments.peg_insertion.plots import create_dynamics_plots
from scripts.environments.peg_insertion.reusable_scene import (
    TASK_ID as REUSABLE_TASK_ID,
)
from scripts.environments.peg_insertion.tacil_network_input import build_network_input
from scripts.environments.render_settings import RenderSettings
from scripts.environments.state_bounds import Bound
from scripts.environments.state_space import StateSpace
from scripts.environments.tacil_dynamics_plotter import TacilDynamicsPlotter
from scripts.environments.tacil_network_input import TacilNetworkInputBuilder
from scripts.environments.video_settings import VideoSettings

TASK_ID = "PegInsertionSide-v1"

CONTROL_MODE = "pd_ee_delta_pose"
REWARD_MODE = "normalized_dense"
SIMULATION_BACKEND = "physx_cpu"


def create_base_environment(
    *,
    max_episode_steps: int,
    render_settings: RenderSettings | None = None,
    reuse_scene: bool = True,
) -> Env:
    return CPUGymWrapper(
        env=gymnasium.make(
            id=REUSABLE_TASK_ID if reuse_scene and render_settings is None else TASK_ID,
            obs_mode="none",
            control_mode=CONTROL_MODE,
            reward_mode=REWARD_MODE,
            sim_backend=SIMULATION_BACKEND,
            render_backend="cpu" if render_settings is not None else "none",
            render_mode="rgb_array" if render_settings is not None else None,
            max_episode_steps=max_episode_steps,
            human_render_camera_configs=(
                {}
                if render_settings is None
                else {"width": render_settings.size, "height": render_settings.size}
            ),
        )
    )


class PegInsertionEnvironment:
    def __init__(self) -> None:
        self._state_space = StateSpace(
            dimensionality=STATE_DIMENSIONALITY, angle_indices=()
        )
        self._state_bounds = (
            *(
                Bound(entries=entries, minimum=-2.0, maximum=2.0)
                for entries in (
                    TCP_POSITION_SLICE,
                    PEG_POSITION_SLICE,
                    HOLE_POSITION_SLICE,
                )
            ),
            Bound(
                entries=slice(HOLE_RADIUS_INDEX, HOLE_RADIUS_INDEX + 1),
                minimum=0.0,
                maximum=float("inf"),
            ),
            Bound(
                entries=slice(PEG_HALF_SIZE_SLICE.start, PEG_HALF_SIZE_SLICE.start + 1),
                minimum=0.0,
                maximum=float("inf"),
            ),
        )

        self._gpi_geometry = PegInsertionGpiGeometry()

    @property
    def state_space(self) -> StateSpace:
        return self._state_space

    @property
    def tacil_network_input_builder(self) -> TacilNetworkInputBuilder:
        return build_network_input

    @property
    def tacil_dynamics_plotter(self) -> TacilDynamicsPlotter:
        return create_dynamics_plots

    @property
    def gpi_geometry(self) -> PegInsertionGpiGeometry:
        return self._gpi_geometry

    @property
    def conditions(self) -> tuple[Condition, ...]:
        return ALL_CONDITIONS

    @property
    def checkpoint_selection_metric(self) -> Literal["reward", "success_then_reward"]:
        return "success_then_reward"

    @property
    def video_settings(self) -> VideoSettings:
        return VideoSettings(frames_per_second=20, playback_speed=2.0)

    def describe_condition(self, *, condition: Condition) -> str:
        return SETTINGS_BY_CONDITION[condition].description

    @staticmethod
    def create_gym_environment(
        *, condition: Condition, render_settings: RenderSettings | None
    ) -> Env:
        environment: Env = _SelectedObservationWrapper(
            environment=create_base_environment(
                # The converted demonstrations need about 105-423 steps so 450 is a good limit.
                max_episode_steps=450,
                render_settings=render_settings,
            )
        )

        disturbance_impulse = SETTINGS_BY_CONDITION[condition].disturbance_impulse
        if disturbance_impulse is None:
            return environment

        return DisturbanceWrapper(
            environment=environment, magnitude=disturbance_impulse
        )

    def calculate_termination(
        self,
        *,
        state_batch: npt.NDArray[np.float32],  # (batch_size, STATE_DIMENSIONALITY)
    ) -> npt.NDArray[np.bool_]:  # (batch_size,)
        inside_batch = state_bounds.contains(
            state_batch=state_batch, bounds=self._state_bounds
        )  # (batch_size,)

        termination_batch = ~inside_batch  # (batch_size,)
        if inside_batch.any():
            termination_batch[inside_batch] = reward.calculate_success(
                state_batch=state_batch[inside_batch]
            )
        return termination_batch


# ---- Private --------------------------------------------


class _SelectedObservationWrapper(gymnasium.ObservationWrapper):
    def __init__(self, *, environment: gymnasium.Env) -> None:
        super().__init__(env=environment)
        task = environment.unwrapped
        if not isinstance(task, PegInsertionSideEnv):
            raise TypeError(f"Expected PegInsertionSideEnv, got {type(task).__name__}")
        self._task = task
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(STATE_DIMENSIONALITY,), dtype=np.float32
        )

    def observation(
        self, observation: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:  # (STATE_DIMENSIONALITY,)
        return extract_state_from_environment(peg_insertion_side_env=self._task)
