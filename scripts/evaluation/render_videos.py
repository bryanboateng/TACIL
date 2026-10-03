import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pygame
import torch
import tyro
import wandb
from gymnasium import Env
from imageio import v2 as imageio
from pygame import Surface
from tqdm import tqdm
from wandb import Run

from scripts.environments.condition import Condition
from scripts.environments.environment import Environment
from scripts.environments.registry import EnvironmentName, create_environment
from scripts.environments.render_settings import RenderSettings
from scripts.evaluation.core.rollout import Episode, roll_out_episodes
from scripts.evaluation.core.scoring import pick_representative_episodes
from scripts.evaluation.core.video import (
    FREEZE_SECONDS,
    calculate_video_frames_per_second,
    render_episode_surfaces,
)
from scripts.evaluation.rollouts_artifact import (
    EVALUATION_ROLLOUTS_ARTIFACT_TYPE,
    download_evaluation_rollouts,
)
from scripts.shared import stage_names
from scripts.shared.log import create_logger
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH
from scripts.shared.policy_types import Policy

logger = create_logger()


@dataclass(frozen=True)
class Config:
    run_paths: tuple[str, ...]


def main() -> None:
    config = tyro.cli(f=Config)
    for run_path in config.run_paths:
        _render_videos(run_path=run_path)


# ---- Private --------------------------------------------


def _render_videos(*, run_path: str) -> None:
    entity, project, run_id = run_path.split("/")
    finished_run = wandb.Api().run(path=run_path)
    environment = create_environment(
        name=EnvironmentName[finished_run.config["environment"]]
    )

    with wandb.init(
        entity=entity,
        project=project,
        id=run_id,
        resume="must",
        dir=OUTPUT_DIRECTORY_PATH,
    ) as run:
        episode_records_by_condition_label = download_evaluation_rollouts(
            artifact=run.use_artifact(
                artifact_or_name=f"evaluation-rollouts-{run_id}:latest",
                type=EVALUATION_ROLLOUTS_ARTIFACT_TYPE,
            )
        )
        for condition in environment.conditions:
            logger.info("Rendering %s of %s.", condition.label, run_path)
            _log_video(
                environment=environment,
                policy_label=f"{finished_run.config['policy_name']} ({run.name})",
                condition=condition,
                episode_records=pick_representative_episodes(
                    episode_records=episode_records_by_condition_label[condition.label]
                ),
                run=run,
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
                    f"{stage_names.EVALUATION}/{condition.label}/6_video": wandb.Video(
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
        dynamic_ncols=True,
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


if __name__ == "__main__":
    main()
