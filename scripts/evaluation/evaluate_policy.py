import tempfile
from pathlib import Path

import numpy as np
import pygame
import torch
import wandb
from gymnasium import Env
from imageio import v2 as imageio
from pygame import Surface
from tqdm import tqdm
from wandb import Run

from scripts.evaluation.core.condition import Condition
from scripts.evaluation.core.renderer import RenderSettings
from scripts.evaluation.core.rollout import (
    EpisodeRecord,
    create_environment,
    roll_out_episodes,
)
from scripts.evaluation.core.scoring import (
    RewardStatistics,
    pick_representative_episodes,
    reward_statistics,
    run_scored_episodes,
    success_rate,
)
from scripts.evaluation.core.video import FREEZE_SECONDS, render_episode_surfaces
from scripts.shared.log import create_logger
from scripts.shared.policy_types import Policy, PolicyFactory

logger = create_logger()


# Every key this stage logs starts with it, so a stage run and a pipeline run
# report the same numbers under the same names. wandb groups panels by the part
# before the first slash, so the stage joins the group with a hyphen rather than
# a slash of its own. The sweep optimizes a key built from it, in create_sweep.
STAGE_NAME = "4_evaluation"


def evaluate_policy(
    *, policy_factory: PolicyFactory, policy_label: str, condition: Condition, run: Run
) -> None:
    episode_records = run_scored_episodes(
        policy_factory=policy_factory,
        start_episode_seed=1_000_000,
        episode_count=100,
        condition=condition,
    )

    _write_summary(
        run=run,
        condition=condition,
        reward=reward_statistics(episode_records),
        success_fraction=success_rate(episode_records),
    )
    _log_video(
        policy_label=policy_label,
        condition=condition,
        episode_records=pick_representative_episodes(episode_records=episode_records),
        stage_name=STAGE_NAME,
        run=run,
    )


def _write_summary(
    *, run: Run, condition: Condition, reward: RewardStatistics, success_fraction: float
) -> None:
    run.summary[f"{STAGE_NAME}/{condition.label}/reward_mean"] = reward.mean
    run.summary[f"{STAGE_NAME}/{condition.label}/reward_std"] = reward.std
    run.summary[f"{STAGE_NAME}/{condition.label}/reward_standard_error"] = (
        reward.standard_error
    )
    run.summary[f"{STAGE_NAME}/{condition.label}/episode_count"] = reward.episode_count
    run.summary[f"{STAGE_NAME}/{condition.label}/success_rate"] = success_fraction


@torch.no_grad()
def _log_video(
    *,
    policy_label: str,
    condition: Condition,
    episode_records: dict[int, EpisodeRecord],
    stage_name: str,
    run: Run,
) -> None:
    fps = 60
    render_settings = RenderSettings(size=1_000, supersample_factor=2)
    environment = create_environment(
        condition=condition, render_settings=render_settings
    )
    try:
        with tempfile.TemporaryDirectory() as video_directory:
            video_path = Path(video_directory) / f"{condition.label}.mp4"
            video_writer = imageio.get_writer(
                uri=video_path,
                fps=fps,
                codec="libx264",
                macro_block_size=1,
                ffmpeg_params=["-preset", "veryfast", "-crf", "20"],
            )
            try:
                _write_episode_frames(
                    policy_label=policy_label,
                    environment=environment,
                    condition=condition,
                    episode_records=episode_records,
                    video_writer=video_writer,
                    fps=fps,
                )
            finally:
                video_writer.close()

            run.log(
                data={
                    f"{stage_name}/{condition.label}/video": wandb.Video(
                        data_or_path=str(video_path), format="mp4"
                    )
                }
            )
    finally:
        environment.close()


def _write_episode_frames(
    *,
    policy_label: str,
    environment: Env,
    condition: Condition,
    episode_records: dict[int, EpisodeRecord],
    video_writer,
    fps: int,
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
        for surface in render_episode_surfaces(
            episode_record=replayed_record,
            condition=condition,
            policy_label=policy_label,
            percentile=percentile,
        ):
            frame = _convert_surface_to_frame(surface=surface)
            video_writer.append_data(frame)

        # Freeze on the final annotated frame.
        if frame is not None:
            for _ in range(fps * FREEZE_SECONDS):
                video_writer.append_data(frame)


def _replay(*, episode_record: EpisodeRecord, environment: Env) -> EpisodeRecord:
    (replayed_record,) = roll_out_episodes(
        environments=[environment],
        seeds=[episode_record.seed],
        policy=_create_replay_policy(action_sequences=[episode_record.actions]),
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


def _convert_surface_to_frame(*, surface: Surface) -> np.ndarray:
    return np.transpose(a=pygame.surfarray.array3d(surface=surface), axes=(1, 0, 2))


def _create_replay_policy(*, action_sequences: list[list[np.ndarray]]) -> Policy:
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
