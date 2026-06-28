from enum import Enum, auto

import numpy as np
import pygame
import torch
import wandb
from tqdm import tqdm
from wandb import Run

from scripts.evaluation.core.disturbance.condition import Condition, label
from scripts.evaluation.core.renderer import RENDER_SIZE
from scripts.evaluation.core.rollout import (
    EpisodeResult,
    create_environment,
    roll_out_episode,
)
from scripts.evaluation.core.scoring import SelectedSeeds
from scripts.shared.policy_types import PolicyFactory

FPS = 10
FREEZE_SECONDS = 2

MARGIN = RENDER_SIZE // 100
TEXT_COLOR = (0, 0, 0)
RUNNING_COLOR = (0, 0, 0)
SUCCESS_COLOR = (20, 120, 40)
FAILURE_COLOR = (170, 30, 30)


class Corner(Enum):
    TOP_LEFT = auto()
    TOP_RIGHT = auto()
    BOTTOM_LEFT = auto()
    BOTTOM_RIGHT = auto()


def create_font() -> pygame.font.Font:
    pygame.font.init()
    return pygame.font.SysFont(name="Inter", size=RENDER_SIZE // 40)


def build_annotated_surfaces(
    *,
    episode_result: EpisodeResult,
    font: pygame.font.Font,
    condition_label: str,
    seed: int,
    reward: float | None,
) -> list[pygame.Surface]:
    """One annotated surface per rendered frame: disturbance + episode identity along the
    top, timestep + run state along the bottom.

    When `reward` is given (recorded videos) it is the episode's scored reward, shown
    statically on every frame. When `reward` is None (live preview) there is no scored
    outcome, so each frame shows its own current reward alongside the running maximum."""
    final_state_text, final_state_color = (
        ("Succeeded", SUCCESS_COLOR)
        if episode_result.terminated
        else ("Failed", FAILURE_COLOR)
    )
    last_frame_index = len(episode_result.frames) - 1
    is_live_preview = reward is None

    annotated_surfaces: list[pygame.Surface] = []
    running_max_reward = 0.0
    for step, frame in enumerate(episode_result.frames):
        state_text, state_color = (
            (final_state_text, final_state_color)
            if step == last_frame_index
            else ("Running", RUNNING_COLOR)
        )
        if is_live_preview:
            # frames[0] is the initial reset frame, before any step/reward.
            current_reward = episode_result.rewards[step - 1] if step >= 1 else 0.0
            running_max_reward = max(running_max_reward, current_reward)
            frame_reward, frame_max_reward = current_reward, running_max_reward
        else:
            frame_reward, frame_max_reward = reward, None
        annotated_surfaces.append(
            _annotate_surface(
                surface=_convert_frame_to_surface(frame=frame),
                font=font,
                seed=seed,
                reward=frame_reward,
                max_reward=frame_max_reward,
                condition_label=condition_label,
                step=step,
                state_text=state_text,
                state_color=state_color,
            )
        )
    return annotated_surfaces


@torch.no_grad()
def record_condition(
    *,
    policy_factory: PolicyFactory,
    condition: Condition,
    selected_seeds: SelectedSeeds,
    run: Run,
) -> None:
    """Re-roll the selected seeds with rendering and log an annotated video per group."""
    font = create_font()
    environment = create_environment(condition=condition, should_render=True)
    try:
        for outcome, seed_rewards in [
            ("Failure", selected_seeds.failures),
            ("Success", selected_seeds.successes),
        ]:
            if not seed_rewards:
                continue

            frames = _record_group(
                policy_factory=policy_factory,
                environment=environment,
                condition=condition,
                outcome=outcome,
                seed_rewards=seed_rewards,
                font=font,
            )

            # wandb.Video expects (T, C, H, W)
            run.summary[f"{label(condition)}/video_{outcome}"] = wandb.Video(
                data_or_path=np.stack(frames).transpose(0, 3, 1, 2), fps=FPS
            )
    finally:
        environment.close()


def _record_group(
    *,
    policy_factory: PolicyFactory,
    environment,
    condition: Condition,
    outcome: str,
    seed_rewards: list[tuple[int, float]],
    font: pygame.font.Font,
) -> list[np.ndarray]:
    condition_label = label(condition)

    frames: list[np.ndarray] = []
    for seed, reward in tqdm(seed_rewards, desc=f"{condition_label}/{outcome}", unit="ep"):
        episode_result = roll_out_episode(
            environment=environment,
            policy=policy_factory(seed=seed),
            seed=seed,
            render=True,
        )
        for surface in build_annotated_surfaces(
            episode_result=episode_result,
            font=font,
            condition_label=condition_label,
            seed=seed,
            reward=reward,
        ):
            frames.append(_convert_surface_to_frame(surface=surface))

        # Freeze on the final annotated frame.
        frames.extend([frames[-1]] * (FPS * FREEZE_SECONDS))
    return frames


def _convert_frame_to_surface(*, frame: np.ndarray) -> pygame.Surface:
    return pygame.surfarray.make_surface(frame.transpose(1, 0, 2))


def _convert_surface_to_frame(*, surface: pygame.Surface) -> np.ndarray:
    return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))


def _annotate_surface(
    *,
    surface: pygame.Surface,
    font: pygame.font.Font,
    seed: int,
    reward: float | None,
    max_reward: float | None,
    condition_label: str,
    step: int,
    state_text: str,
    state_color: tuple[int, int, int],
) -> pygame.Surface:
    annotated_surface = surface.copy()
    identity_text = f"seed {seed}"
    if reward is not None:
        identity_text += f"   reward {reward:.2f}"
    if max_reward is not None:
        identity_text += f"   max {max_reward:.2f}"
    # Disturbance + identity on top, live status on the bottom.
    _blit_text_at_corner(
        surface=annotated_surface,
        font=font,
        text=condition_label,
        corner=Corner.TOP_LEFT,
        color=TEXT_COLOR,
    )
    _blit_text_at_corner(
        surface=annotated_surface,
        font=font,
        text=identity_text,
        corner=Corner.TOP_RIGHT,
        color=TEXT_COLOR,
    )
    _blit_text_at_corner(
        surface=annotated_surface,
        font=font,
        text=f"step {step}",
        corner=Corner.BOTTOM_LEFT,
        color=TEXT_COLOR,
    )
    _blit_text_at_corner(
        surface=annotated_surface,
        font=font,
        text=state_text,
        corner=Corner.BOTTOM_RIGHT,
        color=state_color,
    )
    return annotated_surface


def _blit_text_at_corner(
    *,
    surface: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    corner: Corner,
    color: tuple[int, int, int],
) -> None:
    text_surface = font.render(text, True, color)
    right_aligned_x = surface.get_width() - text_surface.get_width() - MARGIN
    bottom_aligned_y = surface.get_height() - text_surface.get_height() - MARGIN
    match corner:
        case Corner.TOP_LEFT:
            destination = (MARGIN, MARGIN)
        case Corner.TOP_RIGHT:
            destination = (right_aligned_x, MARGIN)
        case Corner.BOTTOM_LEFT:
            destination = (MARGIN, bottom_aligned_y)
        case Corner.BOTTOM_RIGHT:
            destination = (right_aligned_x, bottom_aligned_y)
    surface.blit(source=text_surface, dest=destination)
