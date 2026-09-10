from dataclasses import dataclass

import cv2
import gymnasium
import numpy as np
import pygame
import pymunk.pygame_util
from gym_pusht.envs import PushTEnv
from gymnasium.spaces import Box
from matplotlib import pyplot as plt
from pymunk import Body, Vec2d

from scripts.environments.pusht.disturbance import DisturbanceWrapper


@dataclass(frozen=True)
class RenderSettings:
    size: int
    supersample_factor: int


class RecoloredRenderWrapper(gymnasium.Wrapper):
    def __init__(
        self,
        *,
        environment: gymnasium.Env,
        colormap: str,
        render_settings: RenderSettings,
    ) -> None:
        super().__init__(env=environment)

        push_t_environment = environment.unwrapped
        if not isinstance(push_t_environment, PushTEnv):
            raise TypeError(
                f"Expected PushTEnv, got {type(push_t_environment).__name__}"
            )

        self._render_settings = render_settings
        self._push_t_environment = push_t_environment
        self._disturbance_wrapper = _find_disturbance_wrapper(environment=environment)
        self._simulation_size = _get_simulation_size(environment=push_t_environment)

        self._goal_color = _sample_colormap(name=colormap, value=0.0)
        self._background_color = _sample_colormap(name=colormap, value=0.2)
        self._marker_color = _sample_colormap(name=colormap, value=0.8)
        self._block_color = _sample_colormap(name=colormap, value=0.9)
        self._agent_color = _sample_colormap(name=colormap, value=1.0)

    def render(self) -> np.ndarray:
        return self._compose_image(screen=self._draw_scene())

    # ---- Private --------------------------------------------

    def _draw_scene(self) -> pygame.Surface:
        environment = self._push_t_environment

        canvas_size = (
            self._render_settings.size * self._render_settings.supersample_factor
        )
        scale = canvas_size / self._simulation_size

        screen = pygame.Surface(size=(canvas_size, canvas_size))
        screen.fill(color=self._background_color)

        draw_options = pymunk.pygame_util.DrawOptions(surface=screen)
        draw_options.flags = pymunk.pygame_util.DrawOptions.DRAW_SHAPES
        draw_options.transform = pymunk.Transform.scaling(s=scale)

        goal_body: pymunk.Body = environment.get_goal_pose_body(
            pose=environment.goal_pose
        )
        for shape in environment.block.shapes:
            goal_vertices = [
                pymunk.pygame_util.to_pygame(p=goal_vertex * scale, surface=screen)
                for goal_vertex in (
                    goal_body.local_to_world(v=vertex)
                    for vertex in shape.get_vertices()
                )
            ]
            pygame.draw.polygon(
                surface=screen,
                color=self._goal_color,
                points=goal_vertices + [goal_vertices[0]],
            )

        self._apply_shape_colors()
        environment.space.debug_draw(options=draw_options)
        return screen

    def _apply_shape_colors(self) -> None:
        environment = self._push_t_environment
        for shape in environment.agent.shapes:
            shape.color = self._agent_color
        for shape in environment.block.shapes:
            shape.color = self._block_color
        for shape in environment.space.static_body.shapes:
            # Game border
            shape.color = self._background_color

    def _compose_image(self, *, screen: pygame.Surface) -> np.ndarray:
        image = np.frombuffer(
            buffer=pygame.image.tobytes(
                pygame.transform.smoothscale(
                    surface=screen,
                    size=(self._render_settings.size, self._render_settings.size),
                ),
                "RGB",
            ),
            dtype=np.uint8,
        ).reshape(self._render_settings.size, self._render_settings.size, 3)

        image = image.copy()
        self._draw_last_action(image=image)
        self._draw_push(image=image)
        return image

    def _draw_push(self, *, image: np.ndarray) -> None:
        if (
            self._disturbance_wrapper is None
            or not self._disturbance_wrapper.is_pushing
        ):
            return

        push_direction = self._disturbance_wrapper.push_direction

        block: Body = self._push_t_environment.block
        origin_coordinates = self._scale_to_render(
            point=block.local_to_world(v=block.center_of_gravity)
        )
        arrow_length = int(0.16 * self._render_settings.size)
        tip_coordinates = origin_coordinates + push_direction * arrow_length

        cv2.arrowedLine(
            img=image,
            pt1=origin_coordinates.int_tuple,
            pt2=tip_coordinates.int_tuple,
            color=(40, 200, 80),
            thickness=max(1, int(self._render_settings.size / 250)),
            line_type=cv2.LINE_AA,
            tipLength=0.25,
        )

    def _scale_to_render(self, *, point: Vec2d) -> Vec2d:
        return (point / self._simulation_size) * self._render_settings.size

    def _draw_last_action(self, *, image):
        last_action = self._push_t_environment._last_action
        if last_action is not None:
            coordinate = (
                (np.array(object=last_action) / self._simulation_size)
                * self._render_settings.size
            ).astype(np.int32)
            cv2.drawMarker(
                img=image,
                position=coordinate,
                color=(
                    self._marker_color.r,
                    self._marker_color.g,
                    self._marker_color.b,
                ),
                markerType=cv2.MARKER_CROSS,
                markerSize=max(1, int(8 / 96 * self._render_settings.size)),
                thickness=max(1, int(1 / 96 * self._render_settings.size)),
            )


# ---- Private --------------------------------------------


def _find_disturbance_wrapper(
    *, environment: gymnasium.Env
) -> DisturbanceWrapper | None:
    current = environment
    while isinstance(current, gymnasium.Wrapper):
        if isinstance(current, DisturbanceWrapper):
            return current
        current = current.env
    return None


def _get_simulation_size(*, environment: PushTEnv) -> int:
    action_space = environment.action_space
    if not isinstance(action_space, Box):
        raise TypeError(f"Expected Box action space, got {type(action_space).__name__}")
    return int(action_space.high[0])


def _sample_colormap(*, name: str, value: float) -> pygame.Color:
    rgba_color = plt.get_cmap(name)(X=value)
    return pygame.Color(*(int(255 * channel) for channel in rgba_color[:3]))
