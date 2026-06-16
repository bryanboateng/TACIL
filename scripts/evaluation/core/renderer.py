import types

import matplotlib.pyplot as plt
import pygame
import pymunk.pygame_util
from gymnasium import Env


def _cmap_to_pygame(cmap_name: str, val: float) -> pygame.Color:
    rgba = plt.get_cmap(cmap_name)(val)
    return pygame.Color(*(int(255 * c) for c in rgba[:3]))


def setup_renderer(environment: Env, bg_cmap: str) -> None:
    """Patch the environment renderer with a fixed background color.

    Must be called after env.reset() (so agent/block/space exist) and after
    setup_disturber (so disturber colors are set on the initial state).
    """

    goal_color = _cmap_to_pygame(cmap_name=bg_cmap, val=0.0)
    background_color = _cmap_to_pygame(cmap_name=bg_cmap, val=0.3)
    block_color = _cmap_to_pygame(cmap_name=bg_cmap, val=0.9)
    disturber_color = pygame.Color(204, 0, 0)
    agent_color = pygame.Color(0, 0, 0)

    unwrapped = environment.unwrapped

    def _draw(self):
        screen = pygame.Surface((512, 512))
        screen.fill(color=background_color)

        draw_options = pymunk.pygame_util.DrawOptions(surface=screen)

        goal_body = self.get_goal_pose_body(pose=self.goal_pose)
        for shape in self.block.shapes:
            pts = [
                pymunk.pygame_util.to_pygame(
                    p=goal_body.local_to_world(v=v), surface=screen
                )
                for v in shape.get_vertices()
            ]
            pygame.draw.polygon(screen, goal_color, pts + [pts[0]])

        self.space.debug_draw(options=draw_options)
        return screen

    unwrapped._draw = types.MethodType(_draw, unwrapped)
    _set_colors(unwrapped, background_color, block_color, disturber_color, agent_color)

    original_reset = unwrapped.reset

    def _reset_with_colors(self, **kwargs):
        result = original_reset(**kwargs)

        _set_colors(self, background_color, block_color, disturber_color, agent_color)
        return result

    unwrapped.reset = types.MethodType(_reset_with_colors, unwrapped)


def _set_colors(
    unwrapped, background_color: pygame.Color, block_color, disturber_color, agent_color
) -> None:
    for shape in unwrapped.agent.shapes:
        shape.color = agent_color
    for shape in unwrapped.block.shapes:
        shape.color = block_color
    for shape in unwrapped.space.static_body.shapes:
        # Game border
        shape.color = background_color
    if hasattr(unwrapped, "disturber"):
        for shape in unwrapped.disturber.shapes:
            shape.color = disturber_color
