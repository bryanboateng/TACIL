import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
import pygame
import torch
import wandb
from gymnasium import Env
from imageio import v2 as imageio
from pygame import Surface
from tqdm import tqdm
from wandb import Run

from scripts.environments.condition import Condition
from scripts.environments.environment import Environment
from scripts.environments.render_settings import RenderSettings
from scripts.evaluation.config import EvaluationConfig, resolve_evaluation_stage
from scripts.evaluation.core.rollout import Episode, roll_out_episodes
from scripts.evaluation.core.scoring import (
    RewardStatistics,
    ScoringSession,
    SuccessStatistics,
    calculate_reward_statistics,
    calculate_success_statistics,
    pick_representative_episodes,
)
from scripts.evaluation.core.video import (
    FREEZE_SECONDS,
    calculate_video_frames_per_second,
    render_episode_surfaces,
)
from scripts.shared import stage_names
from scripts.shared.log import create_logger
from scripts.shared.policy_types import Policy, PolicyFactory

logger = create_logger()


STAGE_NAME = stage_names.EVALUATION

ALL_CONDITIONS_LABEL = "0_all_conditions"


def evaluate_policy(
    *,
    environment: Environment,
    policy_factory: PolicyFactory,
    policy_label: str,
    config: EvaluationConfig,
    run: Run,
) -> None:
    reward_means: list[float] = []
    stage = resolve_evaluation_stage(stage=config.stage)

    with (
        torch.no_grad(),
        ScoringSession(
            environment=environment,
            parallel_environment_count=config.parallel_environment_count,
        ) as scoring_session,
    ):
        scoring_start = perf_counter()
        for (
            condition,
            episode_records,
        ) in scoring_session.roll_out_episodes_with_the_same_seeds_in_every_condition(
            policy_factory=policy_factory,
            start_episode_seed=stage.start_episode_seed,
            episode_count_per_condition=stage.episode_count_per_condition,
        ):
            run.summary[f"{STAGE_NAME}/{condition.label}/7_scoring_seconds"] = (
                perf_counter() - scoring_start
            )
            logger.info("Scored the policy in %s.", condition.label)

            reward = calculate_reward_statistics(episode_records=episode_records)
            reward_means.append(reward.mean)

            _write_summary(
                run=run,
                condition=condition,
                reward=reward,
                success=calculate_success_statistics(episode_records=episode_records),
            )
            if config.record_videos:
                video_start = perf_counter()
                _log_video(
                    environment=environment,
                    policy_label=policy_label,
                    condition=condition,
                    episode_records=pick_representative_episodes(
                        episode_records=episode_records
                    ),
                    run=run,
                )
                run.summary[f"{STAGE_NAME}/{condition.label}/8_video_seconds"] = (
                    perf_counter() - video_start
                )
            scoring_start = perf_counter()

    run.summary[f"{STAGE_NAME}/{ALL_CONDITIONS_LABEL}/1_reward_mean"] = sum(
        reward_means
    ) / len(reward_means)


# ---- Private --------------------------------------------


def _write_summary(
    *,
    run: Run,
    condition: Condition,
    reward: RewardStatistics,
    success: SuccessStatistics,
) -> None:
    run.summary[f"{STAGE_NAME}/{condition.label}/1_reward_mean"] = reward.mean
    run.summary[f"{STAGE_NAME}/{condition.label}/2_reward_std"] = reward.std
    run.summary[f"{STAGE_NAME}/{condition.label}/3_reward_standard_error"] = (
        reward.standard_error
    )
    run.summary[f"{STAGE_NAME}/{condition.label}/4_success_rate"] = success.rate
    run.summary[f"{STAGE_NAME}/{condition.label}/5_success_standard_error"] = (
        success.standard_error
    )


def _log_video(
    *,
    environment: Environment,
    policy_label: str,
    condition: Condition,
    episode_records: dict[int, Episode],
    run: Run,
) -> None:
    fps = calculate_video_frames_per_second(
        control_frequency=environment.control_frequency,
        playback_speed=environment.video_playback_speed,
    )
    gym_environment = environment.create_gym_environment(
        condition=condition,
        render_settings=RenderSettings(size=1_000, supersample_factor=2),
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
                    gym_environment=gym_environment,
                    condition=condition,
                    episode_records=episode_records,
                    video_writer=video_writer,
                    fps=fps,
                )
            finally:
                video_writer.close()

            run.log(
                data={
                    f"{STAGE_NAME}/{condition.label}/6_video": wandb.Video(
                        data_or_path=str(video_path), format="mp4"
                    )
                }
            )
    finally:
        gym_environment.close()


def _write_episode_frames(
    *,
    policy_label: str,
    environment: Environment,
    gym_environment: Env,
    condition: Condition,
    episode_records: dict[int, Episode],
    video_writer,
    fps: int,
) -> None:
    for percentile, episode_record in tqdm(
        sorted(episode_records.items(), key=lambda item: item[0]),
        desc="Recording",
        unit="episode",
    ):
        replayed_record = _replay_episode_with_rendering(
            episode_record=episode_record, gym_environment=gym_environment
        )

        frame: np.ndarray | None = None
        for surface in render_episode_surfaces(
            episode_record=replayed_record,
            condition_description=environment.describe_condition(condition=condition),
            playback_speed=environment.video_playback_speed,
            policy_label=policy_label,
            percentile=percentile,
        ):
            frame = _convert_surface_to_frame(surface=surface)
            video_writer.append_data(frame)

        # Freeze on the final frame.
        if frame is not None:
            for _ in range(fps * FREEZE_SECONDS):
                video_writer.append_data(frame)


def _convert_surface_to_frame(*, surface: Surface) -> np.ndarray:
    return np.transpose(a=pygame.surfarray.array3d(surface=surface), axes=(1, 0, 2))


def _replay_episode_with_rendering(
    *, episode_record: Episode, gym_environment: Env
) -> Episode:
    with torch.no_grad():
        (replayed_record,) = roll_out_episodes(
            environments=[gym_environment],
            seeds=[episode_record.seed],
            policy=_create_replay_policy(action_sequences=[episode_record.actions]),
            render=True,
        )

    if not np.array_equal(
        a1=replayed_record.final_observation, a2=episode_record.final_observation
    ):
        raise RuntimeError(
            f"Replaying episode {episode_record.seed} did not reproduce it: "
            f"it ended at {np.array2string(a=replayed_record.final_observation)}, "
            f"not {np.array2string(a=episode_record.final_observation)}."
        )
    return replayed_record


def _create_replay_policy(*, action_sequences: list[list[np.ndarray]]) -> Policy:
    step_indices = [0] * len(action_sequences)

    def create_action(
        *, observations_by_episode_index: dict[int, np.ndarray]
    ) -> dict[int, np.ndarray]:
        actions = {}
        for episode_index in observations_by_episode_index.keys():
            action_sequence = action_sequences[episode_index]
            actions[episode_index] = action_sequence[step_indices[episode_index]]
            step_indices[episode_index] += 1
        return actions

    return create_action
