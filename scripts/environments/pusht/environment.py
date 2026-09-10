import gym_pusht  # noqa: F401
import gymnasium
import numpy as np
import pygame
from gymnasium import Env

from scripts.environments.conditions import Condition, DisturbanceLevel, FloorCondition
from scripts.environments.pusht import disturbance, video
from scripts.environments.pusht.gpi_geometry import PushTGpiGeometry
from scripts.environments.pusht.renderer import RecoloredRenderWrapper, RenderSettings
from scripts.environments.pusht.state_space import PushTStateSpace
from scripts.evaluation.core.rollout import EpisodeRecord

_RENDER_SETTINGS = RenderSettings(size=1_000, supersample_factor=2)


class PushTEnvironment:
    """A two dimensional arena where an agent pushes a T shaped block onto a
    target outline."""

    def __init__(self) -> None:
        self._state_space = PushTStateSpace()
        self._gpi_geometry = PushTGpiGeometry()

    @property
    def state_space(self) -> PushTStateSpace:
        return self._state_space

    @property
    def gpi_geometry(self) -> PushTGpiGeometry:
        return self._gpi_geometry

    @property
    def video_frames_per_second(self) -> int:
        return video.get_video_fps()

    def create_gym_environment(self, *, condition: Condition, render: bool) -> Env:
        match condition.floor_condition:
            case FloorCondition.GRIPPY:
                damping = 0.0
            case FloorCondition.SLIPPERY:
                damping = 0.2

        environment = gymnasium.make(
            id="gym_pusht/PushT-v0",
            damping=damping,
            render_mode="rgb_array" if render else None,
        )

        if condition.disturbance_level is not None:
            environment = disturbance.DisturbanceWrapper(
                environment=environment,
                magnitude=_get_disturbance_magnitude(
                    floor_condition=condition.floor_condition,
                    disturbance_level=condition.disturbance_level,
                ),
            )

        if render:
            match condition.floor_condition:
                case FloorCondition.GRIPPY:
                    colormap = "Blues"
                case FloorCondition.SLIPPERY:
                    colormap = "Oranges"
            environment = RecoloredRenderWrapper(
                environment=environment,
                colormap=colormap,
                render_settings=_RENDER_SETTINGS,
            )

        return environment

    def render_episode_frames(
        self,
        *,
        episode_record: EpisodeRecord,
        condition: Condition,
        policy_label: str,
        percentile: int,
    ) -> list[np.ndarray]:
        return [
            _convert_surface_to_frame(surface=surface)
            for surface in video.render_episode_surfaces(
                episode_record=episode_record,
                condition=condition,
                policy_label=policy_label,
                percentile=percentile,
            )
        ]


# ---- Private --------------------------------------------


def _get_disturbance_magnitude(
    *, floor_condition: FloorCondition, disturbance_level: DisturbanceLevel
) -> float:
    match floor_condition:
        case FloorCondition.GRIPPY:
            match disturbance_level:
                case DisturbanceLevel.LOW:
                    return 1_500
                case DisturbanceLevel.HIGH:
                    return 4_500
        case FloorCondition.SLIPPERY:
            match disturbance_level:
                case DisturbanceLevel.LOW:
                    return 24
                case DisturbanceLevel.HIGH:
                    return 72


def _convert_surface_to_frame(*, surface: pygame.Surface) -> np.ndarray:
    return np.transpose(a=pygame.surfarray.array3d(surface=surface), axes=(1, 0, 2))
