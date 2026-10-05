from typing import Literal

import gym_pusht  # noqa: F401
import gymnasium
import numpy as np
from gym_pusht.envs import PushTEnv
from gymnasium import Env
from numpy import typing as npt

from scripts.environments import state_bounds
from scripts.environments.condition import Condition
from scripts.environments.pusht.condition import ALL_CONDITIONS, SETTINGS_BY_CONDITION
from scripts.environments.pusht.disturbance import DisturbanceWrapper
from scripts.environments.pusht.gpi_geometry import PushTGpiGeometry
from scripts.environments.pusht.observation import (
    ANGLE_INDICES,
    ARENA_SIZE,
    POSITION_SLICE,
    STATE_DIMENSIONALITY,
)
from scripts.environments.pusht.plots import create_dynamics_plots
from scripts.environments.pusht.renderer import RecoloredRenderWrapper
from scripts.environments.pusht.reward import calculate_success, create_environment
from scripts.environments.pusht.tacil_network_input import build_network_input
from scripts.environments.render_settings import RenderSettings
from scripts.environments.state_bounds import Bound
from scripts.environments.state_space import StateSpace
from scripts.environments.tacil_dynamics_plotter import TacilDynamicsPlotter
from scripts.environments.tacil_network_input import TacilNetworkInputBuilder


class PushTEnvironment:
    def __init__(self) -> None:
        self._state_space = StateSpace(
            dimensionality=STATE_DIMENSIONALITY, angle_indices=ANGLE_INDICES
        )
        self._state_bounds = (
            Bound(
                entries=POSITION_SLICE,
                minimum=-float(ARENA_SIZE),
                maximum=2.0 * ARENA_SIZE,
            ),
        )
        self._reward_environment = create_environment()

        self._gpi_geometry = PushTGpiGeometry()

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
    def gpi_geometry(self) -> PushTGpiGeometry:
        return self._gpi_geometry

    @property
    def conditions(self) -> tuple[Condition, ...]:
        return ALL_CONDITIONS

    @property
    def checkpoint_selection_metric(self) -> Literal["reward", "success_then_reward"]:
        return "reward"

    @property
    def control_frequency(self) -> int:
        return 10

    @property
    def video_playback_speed(self) -> float:
        return 3.5

    def describe_condition(self, *, condition: Condition) -> str:
        return SETTINGS_BY_CONDITION[condition].description

    @staticmethod
    def create_gym_environment(
        *, condition: Condition, render_settings: RenderSettings | None
    ) -> Env:
        settings = SETTINGS_BY_CONDITION[condition]

        environment: Env = gymnasium.make(
            id="gym_pusht/PushT-v0",
            damping=settings.damping,
            render_mode="rgb_array" if render_settings is not None else None,
        )

        if settings.disturbance_magnitude is not None:
            environment = DisturbanceWrapper(
                environment=environment, magnitude=settings.disturbance_magnitude
            )

        if render_settings is not None:
            environment = RecoloredRenderWrapper(
                environment=environment,
                colormap=settings.colormap,
                render_settings=render_settings,
            )

        return environment

    @staticmethod
    def copy_simulation_state(*, source: Env, target: Env) -> None:
        source_task = _get_task(environment=source)
        target_task = _get_task(environment=target)
        target_task.agent.position = source_task.agent.position
        target_task.block.position = source_task.block.position
        target_task.block.angle = source_task.block.angle

    def calculate_termination(
        self,
        *,
        state_batch: npt.NDArray[np.float32],  # (batch_size, state_dimensionality)
    ) -> npt.NDArray[np.bool_]:  # (batch_size,)
        inside_batch = state_bounds.contains(
            state_batch=state_batch, bounds=self._state_bounds
        )  # (batch_size,)

        termination_batch = ~inside_batch  # (batch_size,)
        if inside_batch.any():
            termination_batch[inside_batch] = calculate_success(
                state_batch=state_batch[inside_batch],
                environment=self._reward_environment,
            )
        return termination_batch


# ---- Private --------------------------------------------


def _get_task(*, environment: Env) -> PushTEnv:
    task = environment.unwrapped
    if not isinstance(task, PushTEnv):
        raise TypeError(f"Expected PushTEnv, got {type(task).__name__}")
    return task
