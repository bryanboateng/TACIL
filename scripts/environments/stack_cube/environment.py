from typing import Literal

import gymnasium
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box
from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
from numpy import typing as npt

from scripts.environments import state_bounds
from scripts.environments.condition import Condition
from scripts.environments.maniskill import base_environment
from scripts.environments.render_settings import RenderSettings
from scripts.environments.stack_cube.condition import (
    ALL_CONDITIONS,
    SETTINGS_BY_CONDITION,
)
from scripts.environments.stack_cube.disturbance import DisturbanceWrapper
from scripts.environments.stack_cube.gpi_geometry import StackCubeGpiGeometry
from scripts.environments.stack_cube.observation import (
    CUBE_A_POSITION_SLICE,
    CUBE_B_POSITION_SLICE,
    STATE_DIMENSIONALITY,
    TCP_POSITION_SLICE,
    extract_state_from_environment,
)
from scripts.environments.stack_cube.plots import create_dynamics_plots
from scripts.environments.stack_cube.success import calculate_success
from scripts.environments.stack_cube.tacil_network_input import build_network_input
from scripts.environments.state_bounds import Bound
from scripts.environments.state_space import StateSpace
from scripts.environments.tacil_dynamics_plotter import TacilDynamicsPlotter
from scripts.environments.tacil_network_input import TacilNetworkInputBuilder

TASK_ID = "StackCube-v1"


def create_base_environment(
    *, episode_step_limit: int, render_settings: RenderSettings | None = None
) -> Env:
    return base_environment.create_base_environment(
        task_id=TASK_ID,
        episode_step_limit=episode_step_limit,
        render_settings=render_settings,
    )


class StackCubeEnvironment:
    def __init__(self) -> None:
        self._state_space = StateSpace(
            dimensionality=STATE_DIMENSIONALITY, angle_indices=()
        )
        self._state_bounds = tuple(
            Bound(entries=entries, minimum=-2.0, maximum=2.0)
            for entries in (
                TCP_POSITION_SLICE,
                CUBE_A_POSITION_SLICE,
                CUBE_B_POSITION_SLICE,
            )
        )

        self._gpi_geometry = StackCubeGpiGeometry()

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
    def gpi_geometry(self) -> StackCubeGpiGeometry:
        return self._gpi_geometry

    @property
    def conditions(self) -> tuple[Condition, ...]:
        return ALL_CONDITIONS

    @property
    def checkpoint_selection_metric(self) -> Literal["reward", "success_then_reward"]:
        return "success_then_reward"

    @property
    def control_frequency(self) -> int:
        return 20

    @property
    def video_playback_speed(self) -> float:
        return 2.0

    def describe_condition(self, *, condition: Condition) -> str:
        return SETTINGS_BY_CONDITION[condition].description

    @staticmethod
    def create_gym_environment(
        *, condition: Condition, render_settings: RenderSettings | None
    ) -> Env:
        environment: Env = _StateObservationWrapper(
            environment=create_base_environment(
                # Demos take up to ~137 steps (p99), doubled for slack.
                episode_step_limit=300,
                render_settings=render_settings,
            )
        )

        disturbance_shift = SETTINGS_BY_CONDITION[condition].disturbance_shift
        if disturbance_shift is None:
            return environment

        return DisturbanceWrapper(environment=environment, shift=disturbance_shift)

    @staticmethod
    def copy_simulation_state(*, source: Env, target: Env) -> None:
        _get_task(environment=target).set_state_dict(
            _get_task(environment=source).get_state_dict()
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
            termination_batch[inside_batch] = calculate_success(
                state_batch=state_batch[inside_batch]
            )
        return termination_batch


# ---- Private --------------------------------------------


def _get_task(*, environment: Env) -> StackCubeEnv:
    task = environment.unwrapped
    if not isinstance(task, StackCubeEnv):
        raise TypeError(f"Expected StackCubeEnv, got {type(task).__name__}")
    return task


class _StateObservationWrapper(gymnasium.ObservationWrapper):
    def __init__(self, *, environment: gymnasium.Env) -> None:
        super().__init__(env=environment)
        task = environment.unwrapped
        if not isinstance(task, StackCubeEnv):
            raise TypeError(f"Expected StackCubeEnv, got {type(task).__name__}")
        self._task = task
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(STATE_DIMENSIONALITY,), dtype=np.float32
        )

    def observation(
        self, observation: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:  # (STATE_DIMENSIONALITY,)
        return extract_state_from_environment(stack_cube_env=self._task)
