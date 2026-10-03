import tempfile
from pathlib import Path

import numpy as np
import wandb
from wandb import Run

from scripts.evaluation.core.rollout import Episode

EVALUATION_ROLLOUTS_ARTIFACT_TYPE = "evaluation_rollouts"


def upload_evaluation_rollouts(
    *, episode_records_by_condition_label: dict[str, list[Episode]], run: Run
) -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        directory_path = Path(temporary_directory)
        for label, episode_records in episode_records_by_condition_label.items():
            _save_condition_rollouts(
                episode_records=episode_records,
                file_path=directory_path / f"{label}.npz",
            )

        artifact = wandb.Artifact(
            name=f"evaluation-rollouts-{run.id}", type=EVALUATION_ROLLOUTS_ARTIFACT_TYPE
        )
        artifact.add_dir(local_path=str(directory_path))
        run.log_artifact(artifact_or_path=artifact)


def download_evaluation_rollouts(
    *, artifact: wandb.Artifact
) -> dict[str, list[Episode]]:
    directory_path = Path(artifact.download())
    return {
        file_path.stem: _load_condition_rollouts(file_path=file_path)
        for file_path in sorted(directory_path.glob("*.npz"))
    }


# ---- Private --------------------------------------------


def _save_condition_rollouts(
    *, episode_records: list[Episode], file_path: Path
) -> None:
    np.savez_compressed(
        file=file_path,
        seeds=np.array([record.seed for record in episode_records]),
        rewards=np.array([record.reward for record in episode_records]),
        terminated=np.array([record.terminated for record in episode_records]),
        final_observations=np.stack(
            [record.final_observation for record in episode_records]
        ),  # (episode_count, observation_dimensionality)
        step_counts=np.array([len(record.actions) for record in episode_records]),
        actions=np.concatenate(
            [np.stack(record.actions) for record in episode_records]
        ),  # (step_count, action_dimensionality)
        step_rewards=np.concatenate(
            [np.array(record.rewards) for record in episode_records]
        ),  # (step_count,)
    )


def _load_condition_rollouts(*, file_path: Path) -> list[Episode]:
    with np.load(file=file_path) as rollouts:
        split_indices = np.cumsum(rollouts["step_counts"])[:-1]
        action_sequences = np.split(rollouts["actions"], split_indices)
        step_reward_sequences = np.split(rollouts["step_rewards"], split_indices)
        return [
            Episode(
                seed=int(seed),
                reward=float(reward),
                terminated=bool(terminated),
                final_observation=final_observation,
                actions=list(actions),
                rewards=[float(step_reward) for step_reward in step_rewards],
            )
            for seed, reward, terminated, final_observation, actions, step_rewards in zip(
                rollouts["seeds"],
                rollouts["rewards"],
                rollouts["terminated"],
                rollouts["final_observations"],
                action_sequences,
                step_reward_sequences,
                strict=True,
            )
        ]
