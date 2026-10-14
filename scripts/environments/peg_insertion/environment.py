from typing import Literal

import gymnasium

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box
from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv
from numpy import typing as npt

from scripts.environments import state_bounds
from scripts.environments.condition import Condition
from scripts.environments.maniskill import base_environment
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
from scripts.environments.peg_insertion.success import calculate_success
from scripts.environments.peg_insertion.tacil_network_input import build_network_input
from scripts.environments.render_settings import RenderSettings
from scripts.environments.state_bounds import Bound
from scripts.environments.state_space import StateSpace
from scripts.environments.tacil_dynamics_plotter import TacilDynamicsPlotter
from scripts.environments.tacil_network_input import TacilNetworkInputBuilder

TASK_ID = "PegInsertionSide-v1"


def create_base_environment(
    *, episode_step_limit: int, render_settings: RenderSettings | None = None
) -> Env:
    return base_environment.create_base_environment(
        task_id=TASK_ID,
        episode_step_limit=episode_step_limit,
        render_settings=render_settings,
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
                # Demos take up to ~219 steps (p99), doubled for slack.
                episode_step_limit=450,
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


def _get_task(*, environment: Env) -> PegInsertionSideEnv:
    task = environment.unwrapped
    if not isinstance(task, PegInsertionSideEnv):
        raise TypeError(f"Expected PegInsertionSideEnv, got {type(task).__name__}")
    return task


class _StateObservationWrapper(gymnasium.ObservationWrapper):
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
