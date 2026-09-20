from typing import Protocol

import numpy as np
from gymnasium import Env
from numpy import typing as npt

from scripts.environments.condition import Condition
from scripts.environments.gpi_geometry import GpiGeometry
from scripts.environments.network_input import NetworkInputBuilder
from scripts.environments.plot_layout import PlotLayout
from scripts.environments.render_settings import RenderSettings
from scripts.environments.state_space import StateSpace
from scripts.environments.video_settings import VideoSettings


class Environment(Protocol):
    @property
    def state_space(self) -> StateSpace: ...

    @property
    def network_input_builder(self) -> NetworkInputBuilder: ...

    @property
    def plot_layout(self) -> PlotLayout: ...

    @property
    def gpi_geometry(self) -> GpiGeometry: ...

    @property
    def conditions(self) -> tuple[Condition, ...]: ...

    @property
    def video_settings(self) -> VideoSettings: ...

    def describe_condition(self, *, condition: Condition) -> str: ...

    def create_gym_environment(
        self, *, condition: Condition, render_settings: RenderSettings | None
    ) -> Env: ...

    def calculate_termination(
        self,
        *,
        state_batch: npt.NDArray[np.float32],  # (batch_size, state_dimensionality)
    ) -> npt.NDArray[np.bool_]:  # (batch_size,)
        ...
