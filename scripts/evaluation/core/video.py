import tempfile
from enum import Enum, auto
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pygame
import torch
import wandb
from pygame import Surface
from pygame.font import Font
from tqdm import tqdm
from wandb import Run

from scripts.evaluation.core.condition import (
    Condition,
    DisturbanceLevel,
    FloorCondition,
)
from scripts.evaluation.core.renderer import RenderSettings
from scripts.evaluation.core.rollout import (
    EpisodeRecord,
    create_environment,
    roll_out_episodes,
)
from scripts.shared.paths import PROJECT_ROOT_DIRECTORY_PATH
from scripts.shared.policy_types import Policy

FPS = 30
FREEZE_SECONDS = 2
# Multiplier on the font's natural line height, to loosen spacing between
# stacked annotation lines.
LINE_SPACING = 1.25


class Placement(Enum):
    TOP_LEFT = auto()
    TOP_RIGHT = auto()
    BOTTOM_LEFT = auto()
    BOTTOM_CENTER = auto()
    BOTTOM_RIGHT = auto()


def create_font(*, render_size: int) -> pygame.font.Font:
    pygame.font.init()
    return pygame.font.Font(
        PROJECT_ROOT_DIRECTORY_PATH / "assets" / "fonts" / "InterDisplay-Regular.ttf",
        size=render_size // 36,
    )


def build_annotated_surfaces(
    *,
    episode_record: EpisodeRecord,
    condition: Condition,
    policy_label: str,
    notes: str,
    percentile: int,
) -> list[pygame.Surface]:
    if episode_record.frames is None:
        raise ValueError(f"No frames found for episode {episode_record.seed}")

    render_size = episode_record.frames[0].shape[0]
    font = create_font(render_size=render_size)

    last_frame_index = len(episode_record.frames) - 1

    annotated_surfaces: list[pygame.Surface] = []
    running_max_reward = 0.0
    for step, frame in enumerate(episode_record.frames):
        current_reward = episode_record.rewards[step - 1] if step >= 1 else 0.0
        running_max_reward = max(running_max_reward, current_reward)

        annotated_surface = _convert_frame_to_surface(frame=frame).copy()

        _add_text_to_surface(
            text="\n".join(
                line
                for line in (notes, policy_label, _condition_label(condition=condition))
                if line
            ),
            placement=Placement.TOP_LEFT,
            surface=annotated_surface,
            font=font,
        )
        _add_text_to_surface(
            text=(
                f"episode seed: {episode_record.seed}\n"
                f"percentile: {percentile}   score: {episode_record.reward:.2f}"
            ),
            placement=Placement.TOP_RIGHT,
            surface=annotated_surface,
            font=font,
        )
        _add_text_to_surface(
            text=f"step: {step}",
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
                ("Succeeded" if episode_record.terminated else "Failed")
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
    policy_label: str,
    condition: Condition,
    episode_records: dict[int, EpisodeRecord],
    key_prefix: str,
    run: Run,
) -> None:
    render_settings = RenderSettings()
    environment = create_environment(
        condition=condition, render_settings=render_settings
    )
    try:
        with tempfile.TemporaryDirectory() as video_directory:
            video_path = Path(video_directory) / f"{condition.label}.mp4"
            # Stream frames straight into an ffmpeg encoder as they are
            # produced, so encoding overlaps the rollouts and frames never
            # accumulate in memory.
            video_writer = imageio.get_writer(
                uri=video_path,
                fps=FPS,
                codec="libx264",
                macro_block_size=1,
                ffmpeg_params=["-preset", "veryfast", "-crf", "20"],
            )
            try:
                _record_group(
                    policy_label=policy_label,
                    notes=run.notes or "",
                    environment=environment,
                    condition=condition,
                    episode_records=episode_records,
                    video_writer=video_writer,
                )
            finally:
                video_writer.close()

            run.log(
                data={
                    f"{key_prefix}{condition.label}/video": wandb.Video(
                        data_or_path=str(video_path), format="mp4"
                    )
                }
            )
    finally:
        environment.close()


# ---- Private --------------------------------------------


def _condition_label(*, condition: Condition) -> str:
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


def _record_group(
    *,
    policy_label: str,
    notes: str,
    environment,
    condition: Condition,
    episode_records: dict[int, EpisodeRecord],
    video_writer,
) -> None:
    for percentile, episode_record in tqdm(
        sorted(episode_records.items(), key=lambda item: item[0]),
        desc="Recording",
        unit="episode",
    ):
        replayed_record = _replay(
            episode_record=episode_record, environment=environment
        )

        frame: np.ndarray | None = None
        for surface in build_annotated_surfaces(
            episode_record=replayed_record,
            condition=condition,
            policy_label=policy_label,
            notes=notes,
            percentile=percentile,
        ):
            frame = _convert_surface_to_frame(surface=surface)
            video_writer.append_data(frame)

        # Freeze on the final annotated frame.
        if frame is not None:
            for _ in range(FPS * FREEZE_SECONDS):
                video_writer.append_data(frame)


def _replay(*, episode_record: EpisodeRecord, environment) -> EpisodeRecord:
    (replayed_record,) = roll_out_episodes(
        environments=[environment],
        seeds=[episode_record.seed],
        policy=create_replay_policy(action_sequences=[episode_record.actions]),
        render=True,
    )

    if not np.array_equal(
        replayed_record.final_observation, episode_record.final_observation
    ):
        raise RuntimeError(
            f"Replaying episode {episode_record.seed} did not reproduce it: "
            f"it ended at {replayed_record.final_observation}, "
            f"not {episode_record.final_observation}."
        )
    return replayed_record


def _convert_frame_to_surface(*, frame: np.ndarray) -> Surface:
    return pygame.surfarray.make_surface(array=frame.transpose(1, 0, 2))


def _convert_surface_to_frame(*, surface: Surface) -> np.ndarray:
    return np.transpose(a=pygame.surfarray.array3d(surface=surface), axes=(1, 0, 2))


def _add_text_to_surface(
    *, text: str, placement: Placement, surface: Surface, font: Font
) -> None:
    # pygame renders single lines only, so render each line separately and
    # stack them: top placements grow downward from the margin, bottom
    # placements keep their last line on the bottom margin.
    line_surfaces = [font.render(line, True, (0, 0, 0)) for line in text.split("\n")]
    margin = surface.get_width() // 100
    line_height = round(font.get_linesize() * LINE_SPACING)
    block_height = line_height * len(line_surfaces)
    for line_index, line_surface in enumerate(line_surfaces):
        right_aligned_x = surface.get_width() - line_surface.get_width() - margin
        center_aligned_x = (surface.get_width() - line_surface.get_width()) // 2
        top_aligned_y = margin + line_index * line_height
        bottom_aligned_y = (
            surface.get_height() - margin - block_height + line_index * line_height
        )
        match placement:
            case Placement.TOP_LEFT:
                destination = (margin, top_aligned_y)
            case Placement.TOP_RIGHT:
                destination = (right_aligned_x, top_aligned_y)
            case Placement.BOTTOM_LEFT:
                destination = (margin, bottom_aligned_y)
            case Placement.BOTTOM_CENTER:
                destination = (center_aligned_x, bottom_aligned_y)
            case Placement.BOTTOM_RIGHT:
                destination = (right_aligned_x, bottom_aligned_y)
        surface.blit(source=line_surface, dest=destination)


def create_replay_policy(*, action_sequences: list[list[np.ndarray]]) -> Policy:
    """A policy that repeats recorded action sequences, ignoring what it sees.

    The simulator is deterministic, so replaying an episode's actions from its
    seed reproduces it exactly — including the arithmetic of the batch it was
    originally rolled out in, which the policy itself would not reproduce.
    """
    step_indices = [0] * len(action_sequences)

    def policy(*, observations: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
        actions = {}
        for episode_index in observations.keys():
            actions[episode_index] = action_sequences[episode_index][
                step_indices[episode_index]
            ]
            step_indices[episode_index] += 1
        return actions

    return policy
