from enum import Enum, auto

import numpy as np
import pygame
import torch
import wandb
from pygame import Surface
from pygame.font import Font
from tqdm import tqdm
from wandb import Run

from scripts.evaluation.core.condition import Condition, DisturbanceLevel
from scripts.evaluation.core.renderer import RenderSettings
from scripts.evaluation.core.rollout import (
    EpisodeResult,
    create_environment,
    roll_out_episode,
)
from scripts.evaluation.core.scoring import EpisodeScore
from scripts.shared.policy_types import PolicyFactory

FPS = 30
FREEZE_SECONDS = 2


class Placement(Enum):
    TOP_LEFT = auto()
    TOP_RIGHT = auto()
    BOTTOM_LEFT = auto()
    BOTTOM_CENTER = auto()
    BOTTOM_RIGHT = auto()


def create_font(*, render_size: int) -> pygame.font.Font:
    pygame.font.init()
    return pygame.font.SysFont(name="Inter", size=render_size // 40)


def build_annotated_surfaces(
    *,
    episode_result: EpisodeResult,
    font: pygame.font.Font,
    condition: Condition,
    seed: int,
    percentile: int,
) -> list[pygame.Surface]:
    last_frame_index = len(episode_result.frames) - 1

    annotated_surfaces: list[pygame.Surface] = []
    running_max_reward = 0.0
    for step, frame in enumerate(episode_result.frames):
        current_reward = episode_result.rewards[step - 1] if step >= 1 else 0.0
        running_max_reward = max(running_max_reward, current_reward)

        annotated_surface = _convert_frame_to_surface(frame=frame).copy()

        _add_text_to_surface(
            text=_condition_label(condition=condition),
            placement=Placement.TOP_LEFT,
            surface=annotated_surface,
            font=font,
        )
        _add_text_to_surface(
            text=f"P{percentile}   seed {seed}",
            placement=Placement.TOP_RIGHT,
            surface=annotated_surface,
            font=font,
        )
        _add_text_to_surface(
            text=f"step {step}",
            placement=Placement.BOTTOM_LEFT,
            surface=annotated_surface,
            font=font,
        )
        _add_text_to_surface(
            text=_build_reward_text(
                frame_max_reward=running_max_reward, frame_reward=current_reward
            ),
            placement=Placement.BOTTOM_CENTER,
            surface=annotated_surface,
            font=font,
        )
        _add_text_to_surface(
            text=(
                ("Succeeded" if episode_result.terminated else "Failed")
                if step == last_frame_index
                else "Running"
            ),
            placement=Placement.BOTTOM_RIGHT,
            surface=annotated_surface,
            font=font,
        )

        annotated_surfaces.append(annotated_surface)
    return annotated_surfaces


@torch.no_grad()
def record_condition(
    *,
    policy_factory: PolicyFactory,
    condition: Condition,
    episode_scores: dict[int, EpisodeScore],
    run: Run,
) -> None:
    render_settings = RenderSettings()
    font = create_font(render_size=render_settings.size)
    environment = create_environment(
        condition=condition, render_settings=render_settings
    )
    try:
        frames = _record_group(
            policy_factory=policy_factory,
            environment=environment,
            condition=condition,
            episode_scores=episode_scores,
            font=font,
        )

        run.summary[f"{condition.label}/video"] = wandb.Video(
            data_or_path=np.stack(arrays=frames).transpose(0, 3, 1, 2), fps=FPS
        )
    finally:
        environment.close()


# ---- Private --------------------------------------------


def _condition_label(*, condition: Condition) -> str:
    if condition.disturbance is None:
        disturbance_label = "--"
    else:
        match condition.disturbance.level:
            case DisturbanceLevel.LOW:
                level_label = "low"
            case DisturbanceLevel.MEDIUM:
                level_label = "medium"
            case DisturbanceLevel.HIGH:
                level_label = "high"
        disturbance_label = f"{level_label} ({condition.disturbance.magnitude} px/s²)"

    return (
        f"damping: {condition.damping} (vel. kept/s)   disturbance: {disturbance_label}"
    )


def _build_reward_text(*, frame_max_reward: float, frame_reward: float) -> str:
    return f"reward: {frame_reward:.2f}   max: {frame_max_reward:.2f}"


def _record_group(
    *,
    policy_factory: PolicyFactory,
    environment,
    condition: Condition,
    episode_scores: dict[int, EpisodeScore],
    font: pygame.font.Font,
) -> list[np.ndarray]:

    frames: list[np.ndarray] = []
    for percentile, score in tqdm(
        episode_scores.items(), desc=condition.label, unit="episode"
    ):
        episode_result = roll_out_episode(
            environment=environment,
            policy=policy_factory(seed=score.seed),
            seed=score.seed,
            render=True,
        )
        for surface in build_annotated_surfaces(
            episode_result=episode_result,
            font=font,
            condition=condition,
            seed=score.seed,
            percentile=percentile,
        ):
            frames.append(_convert_surface_to_frame(surface=surface))

        # Freeze on the final annotated frame.
        frames.extend([frames[-1]] * (FPS * FREEZE_SECONDS))
    return frames


def _convert_frame_to_surface(*, frame: np.ndarray) -> Surface:
    return pygame.surfarray.make_surface(array=frame.transpose(1, 0, 2))


def _convert_surface_to_frame(*, surface: Surface) -> np.ndarray:
    return np.transpose(a=pygame.surfarray.array3d(surface=surface), axes=(1, 0, 2))


def _add_text_to_surface(
    *, text: str, placement: Placement, surface: Surface, font: Font
) -> None:
    text_surface = font.render(text, True, (0, 0, 0))
    margin = surface.get_width() // 100
    right_aligned_x = surface.get_width() - text_surface.get_width() - margin
    center_aligned_x = (surface.get_width() - text_surface.get_width()) // 2
    bottom_aligned_y = surface.get_height() - text_surface.get_height() - margin
    match placement:
        case Placement.TOP_LEFT:
            destination = (margin, margin)
        case Placement.TOP_RIGHT:
            destination = (right_aligned_x, margin)
        case Placement.BOTTOM_LEFT:
            destination = (margin, bottom_aligned_y)
        case Placement.BOTTOM_CENTER:
            destination = (center_aligned_x, bottom_aligned_y)
        case Placement.BOTTOM_RIGHT:
            destination = (right_aligned_x, bottom_aligned_y)
    surface.blit(source=text_surface, dest=destination)
