from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import pygame
from matplotlib import colormaps
from pygame import Surface
from pygame.font import Font

from scripts.environments.video_settings import VideoSettings
from scripts.evaluation.core.rollout import Episode
from scripts.shared.paths import PROJECT_ROOT_DIRECTORY_PATH

FREEZE_SECONDS = 2


def render_episode_surfaces(
    *,
    episode_record: Episode,
    condition_description: str,
    video_settings: VideoSettings,
    policy_label: str,
    percentile: int,
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
            text=f"{policy_label}\n{condition_description}",
            placement=_Placement.TOP_LEFT,
            surface=annotated_surface,
            font=font,
        )
        _add_text_to_surface(
            text=f"episode seed: {episode_record.seed}",
            placement=_Placement.TOP_RIGHT,
            surface=annotated_surface,
            font=font,
        )
        _add_score_line_to_surface(
            percentile=percentile,
            score=episode_record.reward,
            surface=annotated_surface,
            font=font,
            line_index=1,
        )
        _add_text_to_surface(
            text=f"speed: {video_settings.playback_speed}×\nstep: {step}",
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


def _add_score_line_to_surface(
    *, percentile: int, score: float, surface: Surface, font: Font, line_index: int
) -> None:
    line = _render_boxed_line(
        segments=[
            _Segment(text=f"percentile: {percentile}    score: "),
            _Segment(text=f"{score:.2f}", color=_get_score_color(score=score)),
        ],
        font=font,
    )

    margin = surface.get_width() // 100
    line_height = round(font.get_linesize() * 1.25)
    surface.blit(
        source=line,
        dest=(
            surface.get_width() - line.get_width() - margin,
            margin + line_index * line_height + (line_height - line.get_height()) // 2,
        ),
    )


# ---- Private --------------------------------------------


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


@dataclass(frozen=True)
class _Segment:
    text: str
    color: tuple[int, int, int] | None = None


def _render_tabular_number(
    *, text: str, font: Font, color: tuple[int, int, int]
) -> Surface:
    digit_width = _measure_digit_width(font=font)
    widths = [
        digit_width if character.isdigit() else font.size(character)[0]
        for character in text
    ]
    number = pygame.Surface((sum(widths), font.get_height()), flags=pygame.SRCALPHA)
    offset = 0
    for character, width in zip(text, widths):
        glyph = font.render(character, True, color)
        number.blit(source=glyph, dest=(offset + (width - glyph.get_width()) // 2, 0))
        offset += width
    return number


def _render_boxed_line(*, segments: list[_Segment], font: Font) -> Surface:
    padding = max(2, font.get_height() // 6)
    height = font.get_height() + 2 * padding

    contents: list[tuple[_Segment, Surface]] = []
    for segment in segments:
        if segment.color is None:
            glyphs = font.render(segment.text, True, (0, 0, 0))
        else:
            luminance = (
                0.299 * segment.color[0]
                + 0.587 * segment.color[1]
                + 0.114 * segment.color[2]
            )
            glyphs = _render_tabular_number(
                text=segment.text,
                font=font,
                color=(0, 0, 0) if luminance > 140 else (255, 255, 255),
            )
        contents.append((segment, glyphs))

    line = pygame.Surface(
        (
            sum(
                glyphs.get_width() + (2 * padding if segment.color is not None else 0)
                for segment, glyphs in contents
            ),
            height,
        ),
        flags=pygame.SRCALPHA,
    )

    offset = 0
    for segment, glyphs in contents:
        if segment.color is None:
            line.blit(source=glyphs, dest=(offset, padding))
            offset += glyphs.get_width()
            continue

        box = pygame.Rect(offset, 0, glyphs.get_width() + 2 * padding, height)
        pygame.draw.rect(surface=line, color=segment.color, rect=box)
        line.blit(source=glyphs, dest=(offset + padding, padding))
        offset += box.width
    return line


def _get_score_color(*, score: float) -> tuple[int, int, int]:
    value = min(max(score, 0.0), 1.0)
    red, green, blue, _ = colormaps["RdYlGn"](value)
    return round(255 * red), round(255 * green), round(255 * blue)


def _measure_digit_width(*, font: Font) -> int:
    return max(font.size(digit)[0] for digit in "0123456789")


def _build_reward_text(*, frame_max_reward: float, frame_reward: float) -> str:
    return f"reward: {frame_reward:.2f}   max: {frame_max_reward:.2f}"


def _convert_frame_to_surface(*, frame: np.ndarray) -> Surface:
    return pygame.surfarray.make_surface(array=frame.transpose(1, 0, 2))


def _add_text_to_surface(
    *, text: str, placement: _Placement, surface: Surface, font: Font
) -> None:
    line_surfaces = [font.render(line, True, (0, 0, 0)) for line in text.split("\n")]
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
