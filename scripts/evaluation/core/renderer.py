import cv2
import gymnasium
import numpy as np
import pygame
import pymunk.pygame_util
from gym_pusht.envs import PushTEnv
from gymnasium.spaces import Box
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
from pymunk import Vec2d

from scripts.evaluation.core.disturbance.block_push import BlockPushWrapper

# Defaults for full-quality video recording. The live preview overrides these with a
# smaller render size and no supersampling, since smoothscaling the supersampled canvas
# dominates the per-frame cost.
RENDER_SIZE = 1000
SUPERSAMPLE_FACTOR = 2

# Push visualization: an arrow from the block's center of gravity along the applied push,
# and a dashed approach-axis line through the block (the push is lateral, perpendicular to
# the approach axis). Sizes are derived from the render size at draw time so they scale with
# whichever resolution is in use.
_PUSH_COLOR = (40, 200, 80)  # green: the applied push
_APPROACH_AXIS_COLOR = (70, 110, 200)  # blue: the agent's approach axis


class RecoloredRenderWrapper(gymnasium.Wrapper):
    def __init__(
        self,
        environment: gymnasium.Env,
        colormap: str,
        render_size: int = RENDER_SIZE,
        supersample_factor: int = SUPERSAMPLE_FACTOR,
    ) -> None:
        super().__init__(env=environment)

        push_t_environment = environment.unwrapped
        if not isinstance(push_t_environment, PushTEnv):
            raise TypeError(
                f"Expected PushTEnv, got {type(push_t_environment).__name__}"
            )

        self._render_size = render_size
        self._supersample_factor = supersample_factor
        self._push_t_environment = push_t_environment
        self._block_push_wrapper = _find_block_push_wrapper(environment=environment)
        self._simulation_size = _simulation_size(environment=push_t_environment)

        self._goal_color = _sample_colormap(name=colormap, value=0.0)
        self._background_color = _sample_colormap(name=colormap, value=0.2)
        self._marker_color = _sample_colormap(name=colormap, value=0.8)
        self._block_color = _sample_colormap(name=colormap, value=0.9)
        self._agent_color = _sample_colormap(name=colormap, value=1.0)

    def render(self) -> np.ndarray:
        return self._to_image(screen=self._draw_scene())

    def _draw_scene(self) -> pygame.Surface:
        environment = self._push_t_environment

        canvas_size = self._render_size * self._supersample_factor
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
                    surface=screen, size=(self._render_size, self._render_size)
                ),
                "RGB",
            ),
            dtype=np.uint8,
        ).reshape(self._render_size, self._render_size, 3)
        image = image.copy()
        self.draw_last_action(image)
        self._draw_push(image)
        return image

    def _draw_push(self, image: np.ndarray) -> None:
        if self._block_push_wrapper is None or not self._block_push_wrapper.is_pushing:
            return
        direction = self._block_push_wrapper.push_direction
        # The push is perpendicular to the contact axis, so recover that axis to draw it.
        axis = direction.perpendicular()

        block = self._push_t_environment.block
        origin = self._to_pixel(point=block.local_to_world(v=block.center_of_gravity))

        arrow_length = int(0.16 * self._render_size)
        arrow_thickness = max(2, int(self._render_size / 250))
        thin = max(1, arrow_thickness // 2)
        # Approach axis through the block: the push is perpendicular to it.
        axis_reach = axis * arrow_length
        _draw_dashed_line(
            image=image,
            start=origin - axis_reach,
            end=origin + axis_reach,
            color=_APPROACH_AXIS_COLOR,
            thickness=thin,
        )
        # The applied push.
        tip = origin + direction * arrow_length
        cv2.arrowedLine(
            img=image,
            pt1=_pixel(origin),
            pt2=_pixel(tip),
            color=_PUSH_COLOR,
            thickness=arrow_thickness + 2,
            line_type=cv2.LINE_AA,
            tipLength=0.25,
        )

    def _to_pixel(self, *, point: Vec2d) -> Vec2d:
        return (point / self._simulation_size) * self._render_size

    def draw_last_action(self, image):
        last_action = self._push_t_environment._last_action
        if last_action is not None:
            coordinate = (
                (np.array(object=last_action) / self._simulation_size)
                * self._render_size
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
                markerSize=int(8 / 96 * self._render_size),
                thickness=int(1 / 96 * self._render_size),
            )


def _pixel(point: Vec2d) -> tuple[int, int]:
    return int(point.x), int(point.y)


def _draw_dashed_line(
    *, image: np.ndarray, start: Vec2d, end: Vec2d, color: tuple, thickness: int
) -> None:
    dash = image.shape[0] * 0.018
    span = end - start
    length = span.length
    if length < 1:
        return
    step = dash * 2  # dash followed by an equal gap
    direction = span / length
    offset = 0.0
    while offset < length:
        segment_start = start + direction * offset
        segment_end = start + direction * min(offset + dash, length)
        cv2.line(
            img=image,
            pt1=_pixel(segment_start),
            pt2=_pixel(segment_end),
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
        offset += step


def _find_block_push_wrapper(*, environment: gymnasium.Env) -> BlockPushWrapper | None:
    current = environment
    while isinstance(current, gymnasium.Wrapper):
        if isinstance(current, BlockPushWrapper):
            return current
        current = current.env
    return None


def _simulation_size(*, environment: PushTEnv) -> int:
    action_space = environment.action_space
    if not isinstance(action_space, Box):
        raise TypeError(f"Expected Box action space, got {type(action_space).__name__}")
    return int(action_space.high[0])


def _sample_colormap(*, name: str, value: float) -> pygame.Color:
    rgba_color = plt.get_cmap(name)(X=value)
    return pygame.Color(*(int(255 * channel) for channel in rgba_color[:3]))
