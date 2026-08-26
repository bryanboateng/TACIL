from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import tyro
from rich.pretty import pretty_repr

from scripts.environments.maniskill.observation import (
    POSE_ACTION_POSITION_SLICE,
    has_absolute_pose_actions,
    kept_column_indices,
    known_task_ids,
    robot_root_translation,
)
from scripts.shared.episode_store import EpisodeStore, write_episode_store
from scripts.shared.log import create_logger

logger = create_logger()


@dataclass
class Config:
    task_id: str
    """The ManiSkill task the trajectory belongs to, for example RollBall-v1."""

    trajectory_file_path: Path
    """A ManiSkill trajectory replayed with `--obs-mode state`."""

    output_file_path: Path

    def __post_init__(self):
        if self.task_id not in known_task_ids():
            raise ValueError(
                f"Unknown ManiSkill task {self.task_id!r}. "
                f"Known tasks: {known_task_ids()}"
            )


def main() -> None:
    config = tyro.cli(f=Config)
    logger.info(pretty_repr(config))
    _run(config=config)


def _run(*, config: Config) -> None:
    episode_store = read_maniskill_trajectories(
        trajectory_file_path=config.trajectory_file_path, task_id=config.task_id
    )

    write_episode_store(
        episode_store=episode_store, save_file_path=config.output_file_path
    )

    episode_lengths = np.diff(
        a=episode_store.episode_end_indices, prepend=0
    )  # (episode_count,)

    logger.info(
        "Wrote %d episodes, %d transitions, state %d, action %d",
        len(episode_store.episode_end_indices),
        len(episode_store.states),
        episode_store.states.shape[1],
        episode_store.actions.shape[1],
    )
    logger.info(
        "Episode length: min %d, max %d, mean %.1f",
        episode_lengths.min(),
        episode_lengths.max(),
        episode_lengths.mean(),
    )


def read_maniskill_trajectories(
    *, trajectory_file_path: Path, task_id: str
) -> EpisodeStore:
    # The pipeline works on a subset of the recorded state. The simulator is
    # cut down the same way at rollout time, so both sides match.
    column_indices = kept_column_indices(task_id=task_id)

    # A recorded absolute pose names the gripper position measured from the
    # robot's base, while the state names it measured from the world origin.
    # Everything downstream works in world coordinates, so the recorded
    # actions are moved there once, here, before any normalization.
    action_offset = (
        robot_root_translation(task_id=task_id)
        if has_absolute_pose_actions(task_id=task_id)
        else None
    )

    state_arrays: list[np.ndarray] = []
    action_arrays: list[np.ndarray] = []
    episode_end_indices: list[int] = []
    cumulative_transition_count = 0

    with h5py.File(name=trajectory_file_path, mode="r") as trajectory_file:
        for trajectory_name in _sorted_trajectory_names(
            trajectory_names=trajectory_file.keys()
        ):
            trajectory = trajectory_file[trajectory_name]

            actions = np.asarray(
                a=trajectory["actions"], dtype=np.float32
            )  # (episode_length, action_dimensionality)

            if action_offset is not None:
                actions[:, POSE_ACTION_POSITION_SLICE] += action_offset

            # ManiSkill stores one observation more than actions, because it
            # also records the state the last action leads to. The episode
            # store pairs each state with the action taken in it, so the last
            # observation has no partner and is dropped.
            observations = np.asarray(a=trajectory["obs"], dtype=np.float32)[
                :-1, column_indices
            ]  # (episode_length, state_dimensionality)

            if len(observations) != len(actions):
                raise ValueError(
                    f"{trajectory_name}: {len(observations)} states "
                    f"do not match {len(actions)} actions"
                )

            state_arrays.append(observations)
            action_arrays.append(actions)

            cumulative_transition_count += len(actions)
            episode_end_indices.append(cumulative_transition_count)

    if len(episode_end_indices) == 0:
        raise ValueError(f"No trajectories found in {trajectory_file_path}")

    return EpisodeStore(
        states=np.concatenate(
            state_arrays
        ),  # (total_timestep_count, state_dimensionality)
        actions=np.concatenate(
            action_arrays
        ),  # (total_timestep_count, action_dimensionality)
        episode_end_indices=np.asarray(
            a=episode_end_indices, dtype=np.int64
        ),  # (episode_count,)
    )


# ---- Private --------------------------------------------


def _sorted_trajectory_names(*, trajectory_names: Iterable[str]) -> list[str]:
    """Sort `traj_<n>` names by `n`, because a text sort puts `traj_10` before
    `traj_2`."""
    return sorted(trajectory_names, key=lambda name: int(name.removeprefix("traj_")))


if __name__ == "__main__":
    main()
