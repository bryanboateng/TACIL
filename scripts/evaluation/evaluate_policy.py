import tempfile
from pathlib import Path

import numpy as np
import torch
import wandb
from gymnasium import Env
from imageio import v2 as imageio
from tqdm import tqdm
from wandb import Run

from scripts.environments.conditions import Condition
from scripts.environments.environment import Environment
from scripts.evaluation.core.rollout import EpisodeRecord, roll_out_episodes
from scripts.evaluation.core.scoring import (
    RewardStatistics,
    SuccessStatistics,
    calculate_reward_statistics,
    calculate_success_statistics,
    pick_representative_episodes,
    run_scored_episodes_per_condition,
)
from scripts.shared import stage_names
from scripts.shared.log import create_logger
from scripts.shared.policy_types import Policy, PolicyFactory

logger = create_logger()


STAGE_NAME = stage_names.EVALUATION

ALL_CONDITIONS_LABEL = "0_all_conditions"

# How long the last frame of an episode is held before the next one starts.
_FREEZE_SECONDS = 2


def evaluate_policy(
    *,
    policy_factory: PolicyFactory,
    policy_label: str,
    environment: Environment,
    seed: int,
    run: Run,
) -> None:
    start_episode_seed = 2_000_000 + seed * 10_000

    reward_means: list[float] = []
    with torch.no_grad():
        scored_episodes_per_condition = run_scored_episodes_per_condition(
            policy_factory=policy_factory,
            environment=environment,
            start_episode_seed=start_episode_seed,
            episode_count_per_condition=100,
        )
        for condition, episode_records in scored_episodes_per_condition:
            logger.info("Scored the policy in %s.", condition.label)

            reward = calculate_reward_statistics(episode_records=episode_records)
            reward_means.append(reward.mean)

            _write_summary(
                run=run,
                condition=condition,
                reward=reward,
                success=calculate_success_statistics(episode_records=episode_records),
            )
            _log_video(
                policy_label=policy_label,
                environment=environment,
                condition=condition,
                episode_records=pick_representative_episodes(
                    episode_records=episode_records
                ),
                run=run,
            )

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
    run.summary[f"{STAGE_NAME}/{condition.label}/6_episode_count"] = (
        reward.episode_count
    )


def _log_video(
    *,
    policy_label: str,
    environment: Environment,
    condition: Condition,
    episode_records: dict[int, EpisodeRecord],
    run: Run,
) -> None:
    fps = environment.video_frames_per_second
    gym_environment = environment.create_gym_environment(
        condition=condition, render=True
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
                    f"{STAGE_NAME}/{condition.label}/7_video": wandb.Video(
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
    episode_records: dict[int, EpisodeRecord],
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
        for frame in environment.render_episode_frames(
            episode_record=replayed_record,
            condition=condition,
            policy_label=policy_label,
            percentile=percentile,
        ):
            video_writer.append_data(frame)

        # Freeze on the final frame.
        if frame is not None:
            for _ in range(fps * _FREEZE_SECONDS):
                video_writer.append_data(frame)


def _replay_episode_with_rendering(
    *, episode_record: EpisodeRecord, gym_environment: Env
) -> EpisodeRecord:
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
