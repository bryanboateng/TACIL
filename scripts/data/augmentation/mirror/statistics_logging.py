import numpy as np
from wandb import Run

from scripts.shared.episode_store import EpisodeStore

STAGE_NAME = "2_data"


def log_dataset_statistics(
    *,
    expert_episodes: EpisodeStore,
    mirrored_episodes: EpisodeStore,
    goal_pose_drift: float,
    run: Run,
) -> None:
    # Zero says the reflection leaves the goal exactly where it is, which is
    # what makes the mirrored actions correct labels rather than approximate
    # ones.
    run.summary[f"{STAGE_NAME}/1_transform/goal_pose_drift"] = goal_pose_drift

    # How far the transform actually moves a state. A value near zero would
    # mean the mirrored data repeats the expert data and teaches nothing new.
    run.summary[f"{STAGE_NAME}/1_transform/median_state_displacement"] = float(
        np.median(
            np.linalg.norm(
                mirrored_episodes.states[:, :4] - expert_episodes.states[:, :4], axis=1
            )
        )
    )

    expert_transition_count = int(expert_episodes.states.shape[0])
    mirrored_transition_count = int(mirrored_episodes.states.shape[0])
    total_count = expert_transition_count + mirrored_transition_count

    run.summary[f"{STAGE_NAME}/5_dataset/1_recorded_expert_transitions"] = (
        expert_transition_count
    )
    run.summary[f"{STAGE_NAME}/5_dataset/2_invented_transitions"] = (
        mirrored_transition_count
    )
    run.summary[f"{STAGE_NAME}/5_dataset/4_total_transitions"] = total_count
    run.summary[f"{STAGE_NAME}/5_dataset/5_invented_fraction"] = (
        mirrored_transition_count / total_count
    )
    run.summary[f"{STAGE_NAME}/5_dataset/6_episode_count"] = len(
        expert_episodes.episode_end_indices
    ) + len(mirrored_episodes.episode_end_indices)
