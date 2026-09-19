import gym_pusht  # noqa: F401
import gymnasium
import numpy as np
from gymnasium import Env
from numpy import typing as npt

from scripts.environments.condition import Condition
from scripts.environments.pusht import reward
from scripts.environments.pusht.condition import ALL_CONDITIONS, SETTINGS_BY_CONDITION
from scripts.environments.pusht.disturbance import DisturbanceWrapper
from scripts.environments.pusht.gpi_geometry import PushTGpiGeometry
from scripts.environments.pusht.renderer import RecoloredRenderWrapper
from scripts.environments.pusht.network_input import create_network_input
from scripts.environments.pusht.plot_layout import create_plot_layout
from scripts.environments.pusht.state_space import create_state_space
from scripts.environments.render_settings import RenderSettings
from scripts.environments.network_input import NetworkInput
from scripts.environments.plot_layout import PlotLayout
from scripts.environments.state_space import StateSpace
from scripts.environments.video_settings import VideoSettings
from scripts.shared.episode_store import EpisodeStore

_VIDEO_SETTINGS = VideoSettings(frames_per_second=10, playback_speed=3.5)


class PushTEnvironment:
    def __init__(self) -> None:
        self._state_space = create_state_space()
        self._network_input = create_network_input()
        self._plot_layout = create_plot_layout()
        self._gpi_geometry = PushTGpiGeometry()

    @property
    def state_space(self) -> StateSpace:
        return self._state_space

    @property
    def network_input(self) -> NetworkInput:
        return self._network_input

    @property
    def plot_layout(self) -> PlotLayout:
        return self._plot_layout

    @property
    def gpi_geometry(self) -> PushTGpiGeometry:
        return self._gpi_geometry

    @property
    def conditions(self) -> tuple[Condition, ...]:
        return ALL_CONDITIONS

    @property
    def video_settings(self) -> VideoSettings:
        return _VIDEO_SETTINGS

    def describe_condition(self, *, condition: Condition) -> str:
        return SETTINGS_BY_CONDITION[condition].description

    def create_gym_environment(
        self, *, condition: Condition, render_settings: RenderSettings | None
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

    def calculate_transition_rewards(
        self,
        *,
        episode_store: EpisodeStore,
        transition_indices: npt.NDArray[np.int64],  # (transition_count,)
    ) -> npt.NDArray[np.float32]:  # (transition_count,)
        return reward.calculate_reward(
            state_batch=episode_store.states[transition_indices + 1]
        ).astype(np.float32)

    def calculate_termination(
        self,
        *,
        state_batch: npt.NDArray[np.float32],  # (batch_size, state_dimensionality)
    ) -> npt.NDArray[np.bool_]:  # (batch_size,)
        inside_batch = self._state_space.contains(
            state_batch=state_batch
        )  # (batch_size,)

        termination_batch = ~inside_batch  # (batch_size,)
        if inside_batch.any():
            termination_batch[inside_batch] = reward.calculate_success(
                state_batch=state_batch[inside_batch]
            )
        return termination_batch
