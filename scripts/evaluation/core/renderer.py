import types

import matplotlib.pyplot as plt
import pygame
import pymunk.pygame_util
from gymnasium import Env

_cmap = plt.get_cmap("Grays")


def _mpl_to_pygame(val: float) -> pygame.Color:
    rgba = _cmap(val)
    return pygame.Color(*(int(255 * c) for c in rgba[:3]))


def setup_renderer(environment: Env, bg_color: tuple[int, int, int]) -> None:
    """Patch the environment renderer with a fixed background color.

    Must be called after env.reset() (so agent/block/space exist) and after
    setup_disturber (so disturber colors are set on the initial state).
    """
    unwrapped = environment.unwrapped
    bg = pygame.Color(*bg_color)

    def _draw(self):
        screen = pygame.Surface((512, 512))
        screen.fill(bg)

        draw_options = pymunk.pygame_util.DrawOptions(surface=screen)

        goal_body = self.get_goal_pose_body(pose=self.goal_pose)
        for shape in self.block.shapes:
            pts = [
                pymunk.pygame_util.to_pygame(
                    p=goal_body.local_to_world(v=v), surface=screen
                )
                for v in shape.get_vertices()
            ]
            pygame.draw.polygon(screen, pygame.Color(255, 255, 255), pts + [pts[0]])

        self.space.debug_draw(options=draw_options)
        return screen

    unwrapped._draw = types.MethodType(_draw, unwrapped)
    _set_colors(unwrapped, bg)

    original_reset = unwrapped.reset

    def _reset_with_colors(self, **kwargs):
        result = original_reset(**kwargs)
        _set_colors(self, bg)
        return result

    unwrapped.reset = types.MethodType(_reset_with_colors, unwrapped)


def _set_colors(unwrapped, bg: pygame.Color) -> None:
    for shape in unwrapped.agent.shapes:
        shape.color = pygame.Color(0, 0, 0)
    for shape in unwrapped.block.shapes:
        shape.color = _mpl_to_pygame(0.65)
    for shape in unwrapped.space.static_body.shapes:
        shape.color = bg
    if hasattr(unwrapped, "disturber"):
        for shape in unwrapped.disturber.shapes:
            shape.color = pygame.Color(210, 55, 55)
