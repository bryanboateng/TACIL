import types

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pygame
import pymunk.pygame_util
from gym_pusht.envs import PushTEnv

RENDER_SIZE = 512


def _sample_color_map(name: str, value: float) -> pygame.Color:
    rgba_color = plt.get_cmap(name)(X=value)
    return pygame.Color(*(int(255 * channel) for channel in rgba_color[:3]))


def setup_renderer(*, environment: PushTEnv, color_map: str) -> None:
    goal_color = _sample_color_map(name=color_map, value=0.0)
    background_color = _sample_color_map(name=color_map, value=0.3)
    marker_color = _sample_color_map(name=color_map, value=0.8)
    block_color = _sample_color_map(name=color_map, value=0.9)
    agent_color = _sample_color_map(name=color_map, value=1.0)

    def _draw(self):
        screen = pygame.Surface((RENDER_SIZE, RENDER_SIZE))
        screen.fill(color=background_color)

        draw_options = pymunk.pygame_util.DrawOptions(surface=screen)
        draw_options.flags = pymunk.pygame_util.DrawOptions.DRAW_SHAPES

        goal_body: pymunk.Body = self.get_goal_pose_body(pose=self.goal_pose)
        for shape in self.block.shapes:
            points = []
            for vertex in shape.get_vertices():
                world_point = goal_body.local_to_world(v=vertex)
                points.append(
                    pymunk.pygame_util.to_pygame(
                        p=(world_point.x, world_point.y), surface=screen
                    )
                )
            pygame.draw.polygon(
                surface=screen, color=goal_color, points=points + [points[0]]
            )

        self.space.debug_draw(options=draw_options)
        return screen

    environment._draw = types.MethodType(_draw, environment)  # ty:ignore[unresolved-attribute]
    _set_colors(
        agent_color=agent_color,
        block_color=block_color,
        background_color=background_color,
        environment=environment,
    )

    # gym_pusht paints the action marker as a red cross in `_get_img`. Re-draw it
    # ourselves in the cmap colour instead, on a marker-free base image.
    original_get_img = environment._get_img  # ty:ignore[unresolved-attribute]

    def _get_img_recolored_marker(self, screen, width, height, render_action=False):
        img = original_get_img(screen, width=width, height=height, render_action=False)
        if render_action and self._last_action is not None:
            render_size = min(width, height)
            coord = (np.array(self._last_action) / 512 * [height, width]).astype(
                np.int32
            )
            cv2.drawMarker(
                img,
                coord,
                color=(marker_color.r, marker_color.g, marker_color.b),
                markerType=cv2.MARKER_CROSS,
                markerSize=int(8 / 96 * render_size),
                thickness=int(1 / 96 * render_size),
            )
        return img

    environment._get_img = types.MethodType(_get_img_recolored_marker, environment)  # ty:ignore[unresolved-attribute]

    original_reset = environment.reset

    def _reset_with_colors(self, **kwargs):
        result = original_reset(**kwargs)

        _set_colors(
            agent_color=agent_color,
            block_color=block_color,
            background_color=background_color,
            environment=self,
        )
        return result

    environment.reset = types.MethodType(_reset_with_colors, environment)  # ty:ignore[invalid-assignment]


def _set_colors(
    agent_color: pygame.Color,
    block_color: pygame.Color,
    background_color: pygame.Color,
    environment: PushTEnv,
) -> None:
    for shape in environment.agent.shapes:
        shape.color = agent_color
    for shape in environment.block.shapes:
        shape.color = block_color
    for shape in environment.space.static_body.shapes:
        # Game border
        shape.color = background_color
