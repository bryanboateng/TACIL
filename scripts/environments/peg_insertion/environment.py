import contextlib
from collections.abc import Iterator
from functools import cache
from typing import Any, Literal

import gymnasium
import mani_skill.envs  # noqa: F401
import numpy as np
import sapien
import sapien.wrapper.urdf_loader
from gymnasium import Env
from gymnasium.spaces import Box
from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv
from mani_skill.utils.wrappers import CPUGymWrapper
from numpy import typing as npt

from scripts.environments import state_bounds
from scripts.environments.condition import Condition
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

CONTROL_MODE = "pd_ee_delta_pose"
REWARD_MODE = "normalized_dense"
SIMULATION_BACKEND = "physx_cpu"
# "cuda" pins SAPIEN's renderer to the GPU that CUDA_VISIBLE_DEVICES points to
# instead of relying on SAPIEN's fallback when given "cpu".
# Physics is unaffected and stays on the CPU.
RENDER_DEVICE = "cuda"


class _DummyRenderMaterial:
    """Stand-in for sapien.render.RenderMaterial that never starts a renderer."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass


# Every place that constructs render materials while a scene is built.
# SAPIEN's URDF loader imported the class by name, so it needs its own entry.
_RENDER_MATERIAL_OWNERS = (
    sapien.render,
    sapien.pysapien.render,
    sapien.wrapper.urdf_loader,
)


@contextlib.contextmanager
def dummy_render_materials() -> Iterator[None]:
    """Build scenes without starting SAPIEN's Vulkan renderer.

    ManiSkill only adds visuals when rendering is enabled, but the peg task,
    the ground builder and SAPIEN's URDF loader construct render materials
    before that check.
    Constructing one starts the Vulkan renderer, which shows up as a process
    on a GPU even though nothing is ever rendered.
    Physics does not use the materials, so a dummy is enough.
    """
    originals = [owner.RenderMaterial for owner in _RENDER_MATERIAL_OWNERS]
    for owner in _RENDER_MATERIAL_OWNERS:
        owner.RenderMaterial = _DummyRenderMaterial
    try:
        yield
    finally:
        for owner, original in zip(_RENDER_MATERIAL_OWNERS, originals, strict=True):
            owner.RenderMaterial = original


class _RendererlessWrapper(gymnasium.Wrapper):
    """Rebuild the scene on every reset without starting a renderer."""

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        with dummy_render_materials():
            return self.env.reset(seed=seed, options=options)


@cache
def keep_sapien_renderer_alive() -> sapien.render.SapienRenderer:
    """Hold SAPIEN's renderer for the lifetime of the process.

    Only needed for rendering environments.
    Without this reference the renderer is torn down and started again
    for every episode, which makes resets roughly three times slower.
    """
    return sapien.render.SapienRenderer(sapien.Device(RENDER_DEVICE))


def create_base_environment(
    *, episode_step_limit: int, render_settings: RenderSettings | None = None
) -> Env:
    if render_settings is None:
        with dummy_render_materials():
            return _RendererlessWrapper(
                _make_environment(
                    episode_step_limit=episode_step_limit, render_settings=None
                )
            )
    keep_sapien_renderer_alive()
    return _make_environment(
        episode_step_limit=episode_step_limit, render_settings=render_settings
    )


def _make_environment(
    *, episode_step_limit: int, render_settings: RenderSettings | None
) -> Env:
    return CPUGymWrapper(
        env=gymnasium.make(
            id=TASK_ID,
            obs_mode="none",
            control_mode=CONTROL_MODE,
            reward_mode=REWARD_MODE,
            sim_backend=SIMULATION_BACKEND,
            render_backend=RENDER_DEVICE if render_settings is not None else "none",
            render_mode="rgb_array" if render_settings is not None else None,
            max_episode_steps=episode_step_limit,
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
                # The converted demonstrations need about 105-423 steps so 450 is a good limit.
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
