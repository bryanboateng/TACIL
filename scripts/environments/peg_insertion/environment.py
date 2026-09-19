import gymnasium
import mani_skill.envs  # noqa: F401
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box
from mani_skill.utils.wrappers import CPUGymWrapper
from numpy import typing as npt

from scripts.environments.condition import Condition
from scripts.environments.peg_insertion import reward
from scripts.environments.peg_insertion.condition import (
    ALL_CONDITIONS,
    SETTINGS_BY_CONDITION,
)
from scripts.environments.peg_insertion.disturbance import DisturbanceWrapper
from scripts.environments.peg_insertion.gpi_geometry import PegInsertionGpiGeometry
from scripts.environments.peg_insertion.observation import (
    STATE_DIMENSIONALITY,
    convert_observations,
)
from scripts.environments.peg_insertion.network_input import create_network_input
from scripts.environments.peg_insertion.plot_layout import create_plot_layout
from scripts.environments.peg_insertion.state_space import create_state_space
from scripts.environments.render_settings import RenderSettings
from scripts.environments.network_input import NetworkInput
from scripts.environments.plot_layout import PlotLayout
from scripts.environments.state_space import StateSpace
from scripts.environments.video_settings import VideoSettings
from scripts.shared.episode_store import EpisodeStore

TASK_ID = "PegInsertionSide-v1"

CONTROL_MODE = "pd_joint_pos"

MAX_EPISODE_STEPS = 300

_REWARD_MODE = "normalized_dense"

_SIM_BACKEND = "physx_cpu"

_RENDER_BACKEND = "cpu"

_VIDEO_SETTINGS = VideoSettings(frames_per_second=20, playback_speed=2.0)


class PegInsertionEnvironment:
    def __init__(self) -> None:
        self._state_space = create_state_space()
        self._network_input = create_network_input()
        self._plot_layout = create_plot_layout()
        self._gpi_geometry = PegInsertionGpiGeometry()

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
    def gpi_geometry(self) -> PegInsertionGpiGeometry:
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
        environment: Env = _SelectedObservationWrapper(
            environment=CPUGymWrapper(
                gymnasium.make(
                    id=TASK_ID,
                    obs_mode="state",
                    control_mode=CONTROL_MODE,
                    reward_mode=_REWARD_MODE,
                    sim_backend=_SIM_BACKEND,
                    render_backend=_RENDER_BACKEND,
                    render_mode="rgb_array" if render_settings is not None else None,
                    max_episode_steps=MAX_EPISODE_STEPS,
                    human_render_camera_configs=(
                        {}
                        if render_settings is None
                        else {
                            "width": render_settings.size,
                            "height": render_settings.size,
                        }
                    ),
                )
            )
        )

        disturbance_impulse = SETTINGS_BY_CONDITION[condition].disturbance_impulse
        if disturbance_impulse is None:
            return environment

        return DisturbanceWrapper(
            environment=environment, magnitude=disturbance_impulse
        )

    def calculate_transition_rewards(
        self,
        *,
        episode_store: EpisodeStore,
        transition_indices: npt.NDArray[np.int64],  # (transition_count,)
    ) -> npt.NDArray[np.float32]:  # (transition_count,)
        if episode_store.rewards is None:
            raise ValueError(
                f"{TASK_ID} takes its rewards from the recorded episodes, but the "
                f"dataset carries none. Convert the demonstrations again so that "
                f"the reward of every transition is stored alongside it."
            )
        return episode_store.rewards[transition_indices].astype(np.float32)

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


# ---- Private --------------------------------------------


class _SelectedObservationWrapper(gymnasium.ObservationWrapper):
    def __init__(self, *, environment: gymnasium.Env) -> None:
        super().__init__(env=environment)
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(STATE_DIMENSIONALITY,), dtype=np.float32
        )

    def observation(
        self, observation: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:  # (state_dimensionality,)
        return convert_observations(
            raw_observation_batch=np.asarray(a=observation, dtype=np.float32).reshape(
                1, -1
            )
        )[0]
