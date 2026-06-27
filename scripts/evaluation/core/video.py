import numpy as np
import pygame
import torch
import wandb
from tqdm import tqdm
from wandb import Run

from scripts.evaluation.core.condition import EvaluationCondition
from scripts.evaluation.core.renderer import RENDER_SIZE
from scripts.evaluation.core.rollout import (
    EpisodeResult,
    make_environment,
    roll_out_episode,
)
from scripts.evaluation.core.scoring import SelectedSeeds
from scripts.shared.policy_types import PolicyFactory

FPS = 10
FREEZE_SECONDS = 2

MARGIN = 10
TEXT_COLOR = (0, 0, 0)
RUNNING_COLOR = (0, 0, 0)
SUCCESS_COLOR = (20, 120, 40)
FAILURE_COLOR = (170, 30, 30)


def create_font() -> pygame.font.Font:
    pygame.font.init()
    return pygame.font.SysFont(name="Inter", size=RENDER_SIZE // 23)


def annotated_surfaces(
    *,
    result: EpisodeResult,
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
    final_state, final_color = (
        ("Succeeded", SUCCESS_COLOR) if result.terminated else ("Failed", FAILURE_COLOR)
    )
    last_index = len(result.frames) - 1
    live = reward is None

    surfaces: list[pygame.Surface] = []
    running_max = 0.0
    for step, frame in enumerate(result.frames):
        state, state_color = (
            (final_state, final_color)
            if step == last_index
            else ("Running", RUNNING_COLOR)
        )
        if live:
            # frames[0] is the initial reset frame, before any step/reward.
            current = result.rewards[step - 1] if step >= 1 else 0.0
            running_max = max(running_max, current)
            frame_reward, frame_max = current, running_max
        else:
            frame_reward, frame_max = reward, None
        surfaces.append(
            _annotate(
                surface=_frame_to_surface(frame),
                font=font,
                seed=seed,
                reward=frame_reward,
                max_reward=frame_max,
                condition_label=condition_label,
                step=step,
                state=state,
                state_color=state_color,
            )
        )
    return surfaces


@torch.no_grad()
def record_condition(
    *,
    policy_factory: PolicyFactory,
    condition: EvaluationCondition,
    selected_seeds: SelectedSeeds,
    run: Run,
) -> None:
    """Re-roll the selected seeds with rendering and log an annotated video per group."""
    font = create_font()
    environment = make_environment(condition=condition, render=True)
    try:
        for label, group in [
            ("Failure", selected_seeds.failures),
            ("Success", selected_seeds.successes),
        ]:
            if not group:
                continue

            frames = _record_group(
                policy_factory=policy_factory,
                environment=environment,
                condition=condition,
                label=label,
                group=group,
                font=font,
            )

            # wandb.Video expects (T, C, H, W)
            run.summary[f"{condition}/video_{label}"] = wandb.Video(
                data_or_path=np.stack(frames).transpose(0, 3, 1, 2), fps=FPS
            )
    finally:
        environment.close()


def _record_group(
    *,
    policy_factory: PolicyFactory,
    environment,
    condition: EvaluationCondition,
    label: str,
    group: list[tuple[int, float]],
    font: pygame.font.Font,
) -> list[np.ndarray]:
    condition_label = str(condition)

    frames: list[np.ndarray] = []
    for seed, reward in tqdm(group, desc=f"{condition}/{label}", unit="ep"):
        result = roll_out_episode(
            environment=environment,
            policy=policy_factory(seed=seed),
            seed=seed,
            render=True,
        )
        for surface in annotated_surfaces(
            result=result,
            font=font,
            condition_label=condition_label,
            seed=seed,
            reward=reward,
        ):
            frames.append(_surface_to_frame(surface))

        # Freeze on the final annotated frame.
        frames.extend([frames[-1]] * (FPS * FREEZE_SECONDS))
    return frames


def _frame_to_surface(frame: np.ndarray) -> pygame.Surface:
    return pygame.surfarray.make_surface(frame.transpose(1, 0, 2))


def _surface_to_frame(surface: pygame.Surface) -> np.ndarray:
    return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))


def _annotate(
    *,
    surface: pygame.Surface,
    font: pygame.font.Font,
    seed: int,
    reward: float | None,
    max_reward: float | None,
    condition_label: str,
    step: int,
    state: str,
    state_color: tuple[int, int, int],
) -> pygame.Surface:
    out = surface.copy()
    identity = f"seed {seed}"
    if reward is not None:
        identity += f"   reward {reward:.2f}"
    if max_reward is not None:
        identity += f"   max {max_reward:.2f}"
    # Disturbance + identity on top, live status on the bottom.
    _blit_corner(out, font, condition_label, "top-left", TEXT_COLOR)
    _blit_corner(out, font, identity, "top-right", TEXT_COLOR)
    _blit_corner(out, font, f"step {step}", "bottom-left", TEXT_COLOR)
    _blit_corner(out, font, state, "bottom-right", state_color)
    return out


def _blit_corner(
    surface: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    corner: str,
    color: tuple[int, int, int],
) -> None:
    rendered = font.render(text, True, color)
    right = surface.get_width() - rendered.get_width() - MARGIN
    bottom = surface.get_height() - rendered.get_height() - MARGIN
    dest = {
        "top-left": (MARGIN, MARGIN),
        "top-right": (right, MARGIN),
        "bottom-left": (MARGIN, bottom),
        "bottom-right": (right, bottom),
    }[corner]
    surface.blit(rendered, dest)
