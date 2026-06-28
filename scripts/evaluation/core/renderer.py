import cv2
import gymnasium
import matplotlib.pyplot as plt
import numpy as np
import pygame
import pymunk.pygame_util
from gym_pusht.envs import PushTEnv
from gymnasium.spaces import Box

RENDER_SIZE = 1000
SUPERSAMPLE_FACTOR = 2


class RecoloredRenderWrapper(gymnasium.Wrapper):
    def __init__(self, environment: gymnasium.Env, color_map: str) -> None:
        super().__init__(env=environment)

        push_t_environment = environment.unwrapped
        if not isinstance(push_t_environment, PushTEnv):
            raise TypeError(
                f"Expected PushTEnv, got {type(push_t_environment).__name__}"
            )

        self._push_t_environment = push_t_environment
        # The simulation's world extent is fixed by the env, not by us: positions and
        # actions span [0, action_space.high], and the draw canvas must match it.
        self._simulation_size = _simulation_size(environment=push_t_environment)
        self._goal_color = _sample_color_map(name=color_map, value=0.0)
        self._background_color = _sample_color_map(name=color_map, value=0.2)
        self._marker_color = _sample_color_map(name=color_map, value=0.8)
        self._block_color = _sample_color_map(name=color_map, value=0.9)
        self._agent_color = _sample_color_map(name=color_map, value=1.0)

    def render(self) -> np.ndarray:
        return self._to_image(screen=self._draw_scene())

    def _draw_scene(self) -> pygame.Surface:
        environment = self._push_t_environment

        canvas_size = RENDER_SIZE * SUPERSAMPLE_FACTOR
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

    def _to_image(self, *, screen: pygame.Surface) -> np.ndarray:
        image = np.frombuffer(
            buffer=pygame.image.tobytes(
                pygame.transform.smoothscale(
                    surface=screen, size=(RENDER_SIZE, RENDER_SIZE)
                ),
                "RGB",
            ),
            dtype=np.uint8,
        ).reshape(RENDER_SIZE, RENDER_SIZE, 3)
        image = image.copy()
        self.draw_last_action(image)
        return image

    def draw_last_action(self, image):
        last_action = self._push_t_environment._last_action
        if last_action is not None:
            coordinate = (
                (np.array(object=last_action) / self._simulation_size) * RENDER_SIZE
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
                markerSize=int(8 / 96 * RENDER_SIZE),
                thickness=int(1 / 96 * RENDER_SIZE),
            )


def _simulation_size(*, environment: PushTEnv) -> int:
    action_space = environment.action_space
    if not isinstance(action_space, Box):
        raise TypeError(f"Expected Box action space, got {type(action_space).__name__}")
    return int(action_space.high[0])


def _sample_color_map(*, name: str, value: float) -> pygame.Color:
    rgba_color = plt.get_cmap(name)(X=value)
    return pygame.Color(*(int(255 * channel) for channel in rgba_color[:3]))
