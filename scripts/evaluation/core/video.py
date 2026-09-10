"""What the reader sees written on an episode video."""

from enum import Enum, auto

import numpy as np
import pygame
from pygame import Surface
from pygame.font import Font

from scripts.environments.conditions import Condition, DisturbanceLevel, FloorCondition
from scripts.evaluation.core.rollout import EpisodeRecord
from scripts.shared.paths import PROJECT_ROOT_DIRECTORY_PATH

# How long the last frame of an episode is held before the next one starts.
FREEZE_SECONDS = 2


def render_episode_frames(
    *,
    episode_record: EpisodeRecord,
    condition: Condition,
    policy_label: str,
    percentile: int,
    playback_speed: float,
) -> list[np.ndarray]:
    return [
        convert_surface_to_frame(surface=surface)
        for surface in render_episode_surfaces(
            episode_record=episode_record,
            condition=condition,
            policy_label=policy_label,
            percentile=percentile,
            playback_speed=playback_speed,
        )
    ]


def render_episode_surfaces(
    *,
    episode_record: EpisodeRecord,
    condition: Condition,
    policy_label: str,
    percentile: int,
    playback_speed: float,
) -> list[pygame.Surface]:
    if episode_record.frames is None:
        raise ValueError(f"No frames found for episode {episode_record.seed}")

    font = _create_font(render_size=episode_record.frames[0].shape[0])

    last_frame_index = len(episode_record.frames) - 1

    annotated_surfaces: list[pygame.Surface] = []
    running_max_reward = 0.0
    for step, frame in enumerate(iterable=episode_record.frames):
        current_reward = episode_record.rewards[step - 1] if step >= 1 else 0.0
        running_max_reward = max(running_max_reward, current_reward)

        annotated_surface = _convert_frame_to_surface(frame=frame).copy()

        _add_text_to_surface(
            text=f"{policy_label}\n{_describe_condition(condition=condition)}",
            placement=_Placement.TOP_LEFT,
            surface=annotated_surface,
            font=font,
        )
        _add_text_to_surface(
            text=(
                f"episode seed: {episode_record.seed}\n"
                f"percentile: {percentile}   score: {episode_record.reward:.2f}"
            ),
            placement=_Placement.TOP_RIGHT,
            surface=annotated_surface,
            font=font,
        )
        _add_text_to_surface(
            text=f"speed: {_describe_speed(playback_speed=playback_speed)}\nstep: {step}",
            placement=_Placement.BOTTOM_LEFT,
            surface=annotated_surface,
            font=font,
        )
        _add_text_to_surface(
            text=_build_reward_text(
                frame_max_reward=running_max_reward, frame_reward=current_reward
            ),
            placement=_Placement.BOTTOM_CENTER,
            surface=annotated_surface,
            font=font,
        )
        _add_text_to_surface(
            text=(
                ("Succeeded" if episode_record.terminated else "Failed")
                if step == last_frame_index
                else "Running"
            ),
            placement=_Placement.BOTTOM_RIGHT,
            surface=annotated_surface,
            font=font,
        )

        annotated_surfaces.append(annotated_surface)
    return annotated_surfaces


def convert_surface_to_frame(*, surface: Surface) -> np.ndarray:
    return np.transpose(a=pygame.surfarray.array3d(surface=surface), axes=(1, 0, 2))


# ---- Private --------------------------------------------

_TEXT_COLOR = (0, 0, 0)
_TEXT_OUTLINE_COLOR = (255, 255, 255)


def _describe_speed(*, playback_speed: float) -> str:
    """Write a whole speed without its trailing zero, so 1.0 reads as 1."""
    return f"{playback_speed:g}×"


class _Placement(Enum):
    TOP_LEFT = auto()
    TOP_RIGHT = auto()
    BOTTOM_LEFT = auto()
    BOTTOM_CENTER = auto()
    BOTTOM_RIGHT = auto()


def _create_font(*, render_size: int) -> pygame.font.Font:
    pygame.font.init()
    return pygame.font.Font(
        PROJECT_ROOT_DIRECTORY_PATH / "assets" / "fonts" / "InterDisplay-Regular.ttf",
        size=render_size // 36,
    )


def _describe_condition(*, condition: Condition) -> str:
    if condition.disturbance_level is None:
        disturbance_label = "none"
    else:
        match condition.disturbance_level:
            case DisturbanceLevel.LOW:
                disturbance_label = "low"
            case DisturbanceLevel.HIGH:
                disturbance_label = "high"

    match condition.floor_condition:
        case FloorCondition.GRIPPY:
            floor_label = "grippy"
        case FloorCondition.SLIPPERY:
            floor_label = "slippery"

    return f"floor: {floor_label}   disturbance: {disturbance_label}"


def _build_reward_text(*, frame_max_reward: float, frame_reward: float) -> str:
    return f"reward: {frame_reward:.2f}   max: {frame_max_reward:.2f}"


def _convert_frame_to_surface(*, frame: np.ndarray) -> Surface:
    return pygame.surfarray.make_surface(array=frame.transpose(1, 0, 2))


def _render_line(*, text: str, font: Font) -> Surface:
    """Draw one line, dark on a light outline.

    The outline keeps the writing readable over a dark picture. Push T draws
    a light arena, but a ManiSkill camera sees a black background above the
    table.
    """
    outline_width = max(1, font.get_height() // 14)
    core = font.render(text, True, _TEXT_COLOR)
    outline = font.render(text, True, _TEXT_OUTLINE_COLOR)

    line = Surface(
        size=(
            core.get_width() + 2 * outline_width,
            core.get_height() + 2 * outline_width,
        ),
        flags=pygame.SRCALPHA,
    )
    for horizontal_offset in range(-outline_width, outline_width + 1):
        for vertical_offset in range(-outline_width, outline_width + 1):
            line.blit(
                source=outline,
                dest=(
                    outline_width + horizontal_offset,
                    outline_width + vertical_offset,
                ),
            )
    line.blit(source=core, dest=(outline_width, outline_width))
    return line


def _add_text_to_surface(
    *, text: str, placement: _Placement, surface: Surface, font: Font
) -> None:
    line_surfaces = [_render_line(text=line, font=font) for line in text.split("\n")]
    margin = surface.get_width() // 100
    line_height = round(font.get_linesize() * 1.25)
    block_height = line_height * len(line_surfaces)
    for line_index, line_surface in enumerate(line_surfaces):
        right_aligned_x = surface.get_width() - line_surface.get_width() - margin
        center_aligned_x = (surface.get_width() - line_surface.get_width()) // 2
        top_aligned_y = margin + line_index * line_height
        bottom_aligned_y = (
            surface.get_height() - margin - block_height + line_index * line_height
        )
        match placement:
            case _Placement.TOP_LEFT:
                destination = (margin, top_aligned_y)
            case _Placement.TOP_RIGHT:
                destination = (right_aligned_x, top_aligned_y)
            case _Placement.BOTTOM_LEFT:
                destination = (margin, bottom_aligned_y)
            case _Placement.BOTTOM_CENTER:
                destination = (center_aligned_x, bottom_aligned_y)
            case _Placement.BOTTOM_RIGHT:
                destination = (right_aligned_x, bottom_aligned_y)
        surface.blit(source=line_surface, dest=destination)
