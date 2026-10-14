import contextlib
from collections.abc import Iterator
from functools import cache
from typing import Any

import gymnasium
import mani_skill.envs  # noqa: F401
import sapien
import sapien.wrapper.urdf_loader
from gymnasium import Env
from mani_skill.utils.wrappers import CPUGymWrapper

from scripts.environments.render_settings import RenderSettings

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

    ManiSkill only adds visuals when rendering is enabled, but the tasks,
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
    *,
    task_id: str,
    episode_step_limit: int,
    render_settings: RenderSettings | None = None,
) -> Env:
    if render_settings is None:
        with dummy_render_materials():
            return _RendererlessWrapper(
                _make_environment(
                    task_id=task_id,
                    episode_step_limit=episode_step_limit,
                    render_settings=None,
                )
            )
    keep_sapien_renderer_alive()
    return _make_environment(
        task_id=task_id,
        episode_step_limit=episode_step_limit,
        render_settings=render_settings,
    )


# ---- Private --------------------------------------------


def _make_environment(
    *, task_id: str, episode_step_limit: int, render_settings: RenderSettings | None
) -> Env:
    return CPUGymWrapper(
        env=gymnasium.make(
            id=task_id,
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
