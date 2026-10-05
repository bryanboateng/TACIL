from typing import Literal, Protocol

import numpy as np
from gymnasium import Env
from numpy import typing as npt

from scripts.environments.condition import Condition
from scripts.environments.gpi_geometry import GpiGeometry
from scripts.environments.render_settings import RenderSettings
from scripts.environments.state_space import StateSpace
from scripts.environments.tacil_dynamics_plotter import TacilDynamicsPlotter
from scripts.environments.tacil_network_input import TacilNetworkInputBuilder


class Environment(Protocol):
    @property
    def state_space(self) -> StateSpace: ...

    @property
    def tacil_network_input_builder(self) -> TacilNetworkInputBuilder: ...

    @property
    def tacil_dynamics_plotter(self) -> TacilDynamicsPlotter: ...

    @property
    def gpi_geometry(self) -> GpiGeometry: ...

    @property
    def conditions(self) -> tuple[Condition, ...]: ...

    @property
    def checkpoint_selection_metric(
        self,
    ) -> Literal["reward", "success_then_reward"]: ...

    @property
    def control_frequency(self) -> int: ...

    @property
    def video_playback_speed(self) -> float: ...

    def describe_condition(self, *, condition: Condition) -> str: ...

    def create_gym_environment(
        self, *, condition: Condition, render_settings: RenderSettings | None
    ) -> Env: ...

    def copy_simulation_state(self, *, source: Env, target: Env) -> None: ...

    def calculate_termination(
        self,
        *,
        state_batch: npt.NDArray[np.float32],  # (batch_size, state_dimensionality)
    ) -> npt.NDArray[np.bool_]:  # (batch_size,)
        ...
