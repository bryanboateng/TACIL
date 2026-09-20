import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from gymnasium import Env
from numpy import typing as npt
from tqdm import tqdm

from scripts.environments.peg_insertion.condition import UNDISTURBED_CONDITION
from scripts.environments.peg_insertion.environment import (
    CONTROL_MODE,
    TASK_ID,
    PegInsertionEnvironment,
)
from scripts.environments.peg_insertion.observation import ACTION_DIMENSIONALITY
from scripts.shared.episode_store import RewardedEpisodeStore

_TRAJECTORY_FILE_NAME = "trajectory.h5"
_METADATA_FILE_NAME = "trajectory.json"


def build_episode_store(*, source_path: Path) -> RewardedEpisodeStore:
    metadata = json.loads(s=(source_path / _METADATA_FILE_NAME).read_text())
    _check_metadata(metadata=metadata)

    recorded_episodes = metadata["episodes"]

    state_arrays: list[npt.NDArray[np.float32]] = []
    action_arrays: list[npt.NDArray[np.float32]] = []
    reward_arrays: list[npt.NDArray[np.float32]] = []
    episode_end_indices: list[int] = []
    cumulative_transition_count = 0

    with h5py.File(name=source_path / _TRAJECTORY_FILE_NAME, mode="r") as trajectory:
        recorded_action_arrays = [
            np.asarray(
                a=trajectory[f"traj_{recorded_episode['episode_id']}"]["actions"],
                dtype=np.float32,
            )  # (recorded_length, action_dimensionality)
            for recorded_episode in recorded_episodes
        ]

    gym_environment = PegInsertionEnvironment().create_gym_environment(
        condition=UNDISTURBED_CONDITION,
        render_settings=None,
        max_episode_steps=max(len(actions) for actions in recorded_action_arrays),
    )

    try:
        for recorded_episode, recorded_actions in tqdm(
            iterable=list(zip(recorded_episodes, recorded_action_arrays, strict=True)),
            desc="Replaying",
            unit="episode",
        ):
            replayed = _replay_episode(
                gym_environment=gym_environment,
                episode_id=recorded_episode["episode_id"],
                seed=recorded_episode["episode_seed"],
                recorded_actions=recorded_actions,
            )

            state_arrays.append(replayed.states)
            action_arrays.append(replayed.actions)
            reward_arrays.append(replayed.rewards)

            cumulative_transition_count += len(replayed.actions)
            episode_end_indices.append(cumulative_transition_count)
    finally:
        gym_environment.close()

    return RewardedEpisodeStore(
        states=np.concatenate(
            state_arrays
        ),  # (total_timestep_count, state_dimensionality)
        actions=np.concatenate(
            action_arrays
        ),  # (total_timestep_count, action_dimensionality)
        rewards=np.concatenate(reward_arrays),  # (total_timestep_count,)
        episode_end_indices=np.asarray(
            a=episode_end_indices, dtype=np.int64
        ),  # (episode_count,)
    )


# ---- Private --------------------------------------------


@dataclass(frozen=True)
class _ReplayedEpisode:
    states: npt.NDArray[np.float32]  # (episode_length, state_dimensionality)
    actions: npt.NDArray[np.float32]  # (episode_length, action_dimensionality)
    rewards: npt.NDArray[np.float32]  # (episode_length,)


def _check_metadata(*, metadata: dict) -> None:
    environment_info = metadata["env_info"]
    if environment_info["env_id"] != TASK_ID:
        raise ValueError(
            f"The demonstrations were recorded for "
            f"{environment_info['env_id']!r}, not {TASK_ID!r}."
        )

    recorded_control_mode = environment_info["env_kwargs"].get("control_mode")
    if recorded_control_mode != CONTROL_MODE:
        raise ValueError(
            f"The demonstrations were recorded under {recorded_control_mode!r}, but "
            f"the pipeline drives {TASK_ID} with {CONTROL_MODE!r}. Replaying them "
            f"would not reproduce the recorded motion."
        )


def _replay_episode(
    *,
    gym_environment: Env,
    episode_id: int,
    seed: int,
    recorded_actions: npt.NDArray[np.float32],  # (recorded_length, action_count)
) -> _ReplayedEpisode:
    if recorded_actions.shape[1] != ACTION_DIMENSIONALITY:
        raise ValueError(
            f"The demonstration holds {recorded_actions.shape[1]}-entry actions, but "
            f"{TASK_ID} under {CONTROL_MODE} takes {ACTION_DIMENSIONALITY} entries."
        )

    state, _ = gym_environment.reset(seed=seed)

    states: list[npt.NDArray[np.float32]] = []
    rewards: list[float] = []
    solved = False

    for action in recorded_actions:
        states.append(state)

        state, step_reward, terminated, _, _ = gym_environment.step(action)
        rewards.append(float(step_reward))
        solved = solved or bool(terminated)

    if not solved:
        raise RuntimeError(
            f"Replaying demonstration {episode_id} never solved {TASK_ID}, although "
            f"every recorded demonstration ends in success. The replay no longer "
            f"reproduces the recording."
        )

    rewards[-1] = float("nan")

    return _ReplayedEpisode(
        states=np.stack(arrays=states).astype(np.float32),
        actions=recorded_actions,
        rewards=np.asarray(a=rewards, dtype=np.float32),
    )
