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
from scripts.evaluation.core.rollout import Episode
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
from scripts.shared.dataset_artifact import read_episode_count
from scripts.shared.log import create_logger
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH

logger = create_logger()


@dataclass(frozen=True)
class Config:
    video_project: str
    run_paths: tuple[str, ...]


def main() -> None:
    config = tyro.cli(f=Config)
    for run_path in config.run_paths:
        _render_videos(run_path=run_path, video_project=config.video_project)


# ---- Private --------------------------------------------


def _render_videos(*, run_path: str, video_project: str) -> None:
    entity, project, run_id = run_path.split("/")
    source_run = wandb.Api().run(path=run_path)
    environment = create_environment(
        name=EnvironmentName[source_run.config["environment"]]
    )

    with wandb.init(
        entity=entity,
        project=video_project,
        job_type="video",
        name=source_run.name,
        config=dict(source_run.config, source_run_path=run_path),
        dir=OUTPUT_DIRECTORY_PATH,
    ) as run:
        episode_records_by_condition_label = download_evaluation_rollouts(
            artifact=run.use_artifact(
                artifact_or_name=(
                    f"{entity}/{project}/evaluation-rollouts-{run_id}:latest"
                ),
                type=EVALUATION_ROLLOUTS_ARTIFACT_TYPE,
            )
        )
        for condition in environment.conditions:
            logger.info("Rendering %s of %s.", condition.label, run_path)
            _log_video(
                environment=environment,
                policy_label=_describe_run(config=source_run.config),
                condition=condition,
                episode_records=pick_representative_episodes(
                    episode_records=episode_records_by_condition_label[condition.label]
                ),
                run=run,
            )


def _describe_run(*, config: dict) -> str:
    episode_count = read_episode_count(
        dataset_name=config["expert_dataset_artifact"].split(":")[0]
    )
    return f"{_describe_method(config=config)} · {episode_count} Episodes"


def _describe_method(*, config: dict) -> str:
    policy_name = config["policy_name"]
    policy = policy_name.upper()
    augmentation = config.get("augmentation")
    if augmentation is not None:
        return f"{policy} {augmentation['name'].upper()}"
    if (
        policy_name in ("bc", "dp")
        and config["policy"].get("observation_noise") is not None
    ):
        return f"{policy} Noise"
    return policy


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
    physics_environment = environment.create_gym_environment(
        condition=condition, render_settings=None
    )
    render_environment = environment.create_gym_environment(
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
                    physics_environment=physics_environment,
                    render_environment=render_environment,
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
        render_environment.close()
        physics_environment.close()


def _write_episode_frames(
    *,
    policy_label: str,
    environment: Environment,
    physics_environment: Env,
    render_environment: Env,
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
            environment=environment,
            episode_record=episode_record,
            physics_environment=physics_environment,
            render_environment=render_environment,
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
    *,
    environment: Environment,
    episode_record: Episode,
    physics_environment: Env,
    render_environment: Env,
) -> Episode:
    observation, _ = physics_environment.reset(seed=episode_record.seed)
    render_environment.reset(seed=episode_record.seed)
    frames = [
        _render_physics_state(
            environment=environment,
            physics_environment=physics_environment,
            render_environment=render_environment,
        )
    ]
    rewards: list[float] = []
    terminated = False
    with torch.no_grad():
        for action in episode_record.actions:
            observation, reward, terminated, _, _ = physics_environment.step(
                action=action
            )
            rewards.append(float(reward))
            frames.append(
                _render_physics_state(
                    environment=environment,
                    physics_environment=physics_environment,
                    render_environment=render_environment,
                )
            )

    if not np.array_equal(a1=observation, a2=episode_record.final_observation):
        raise RuntimeError(
            f"Replaying episode {episode_record.seed} did not reproduce it: "
            f"it ended at {np.array2string(a=observation)}, "
            f"not {np.array2string(a=episode_record.final_observation)}."
        )
    return Episode(
        seed=episode_record.seed,
        reward=max(rewards, default=float("nan")),
        terminated=terminated,
        final_observation=observation,
        actions=episode_record.actions,
        rewards=rewards,
        frames=frames,
    )


def _render_physics_state(
    *, environment: Environment, physics_environment: Env, render_environment: Env
) -> np.ndarray:
    environment.copy_simulation_state(
        source=physics_environment, target=render_environment
    )
    frame = render_environment.render()
    if not isinstance(frame, np.ndarray):
        raise TypeError(
            f"Expected rgb_array render to return ndarray, got {type(frame)}"
        )
    return frame


if __name__ == "__main__":
    main()
