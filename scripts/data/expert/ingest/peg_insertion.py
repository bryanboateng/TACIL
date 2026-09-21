import json
from dataclasses import dataclass

import h5py
import numpy as np
from gymnasium import Env
from mani_skill.utils.download_demo import DATASET_SOURCES
from numpy import typing as npt
from tqdm import tqdm

from scripts.environments.peg_insertion.condition import UNDISTURBED_CONDITION
from scripts.environments.peg_insertion.environment import (
    CONTROL_MODE,
    TASK_ID,
    PegInsertionEnvironment,
)
from scripts.environments.peg_insertion.observation import (
    ACTION_DIMENSIONALITY,
    convert_observations,
)
from scripts.shared.download import download_and_extract_zip
from scripts.shared.episode_store import RewardedEpisodeStore

_DENSE_REWARD_SCALE = 10.0
_REPLAY_CHECKED_EPISODE_COUNT = 3
_REPLAY_TOLERANCE = 1e-5


@dataclass(frozen=True)
class _Episode:
    states: npt.NDArray[np.float32]  # (episode_length, state_dimensionality)
    actions: npt.NDArray[np.float32]  # (episode_length, action_dimensionality)
    rewards: npt.NDArray[np.float32]  # (episode_length,)


def build_episode_store() -> RewardedEpisodeStore:
    source_path = (
        download_and_extract_zip(url=DATASET_SOURCES[TASK_ID].raw_dataset_url)
        / TASK_ID
        / "motionplanning"
    )

    metadata = json.loads(s=(source_path / "trajectory.json").read_text())
    _check_metadata(metadata=metadata)
    recorded_episodes = metadata["episodes"]

    episodes: list[_Episode] = []

    with h5py.File(name=source_path / "trajectory.h5", mode="r") as trajectory_file:
        gym_environment = PegInsertionEnvironment().create_gym_environment(
            condition=UNDISTURBED_CONDITION,
            render_settings=None,
            max_episode_steps=max(
                len(
                    trajectory_file[f"traj_{recorded_episode['episode_id']}"]["actions"]
                )
                for recorded_episode in recorded_episodes
            ),
        )

        try:
            for position, recorded_episode in enumerate(
                tqdm(iterable=recorded_episodes, desc="Reading", unit="episode")
            ):
                recording = _Recording(
                    group=trajectory_file[f"traj_{recorded_episode['episode_id']}"],
                    seed=recorded_episode["episode_seed"],
                    episode_id=recorded_episode["episode_id"],
                )
                _check_recording(recording=recording)

                episode = _restore_episode(
                    gym_environment=gym_environment, recording=recording
                )
                if position < _REPLAY_CHECKED_EPISODE_COUNT:
                    _check_against_replay(
                        gym_environment=gym_environment,
                        recording=recording,
                        episode=episode,
                    )

                episodes.append(episode)
        finally:
            gym_environment.close()

    return RewardedEpisodeStore(
        states=np.concatenate(
            [episode.states for episode in episodes]
        ),  # (total_timestep_count, state_dimensionality)
        actions=np.concatenate(
            [episode.actions for episode in episodes]
        ),  # (total_timestep_count, action_dimensionality)
        episode_end_indices=np.cumsum(
            a=np.asarray(
                a=[len(episode.actions) for episode in episodes], dtype=np.int64
            )
        ),  # (episode_count,)
        rewards=np.concatenate(
            [episode.rewards for episode in episodes]
        ),  # (total_timestep_count,)
    )


# ---- Private --------------------------------------------


@dataclass(frozen=True)
class _Recording:
    group: h5py.Group
    seed: int
    episode_id: int

    @property
    def actions(self) -> npt.NDArray[np.float32]:  # (episode_length, action_count)
        return np.asarray(a=self.group["actions"], dtype=np.float32)

    @property
    def rewards(self) -> npt.NDArray[np.float32]:  # (episode_length,)
        return np.asarray(a=self.group["rewards"], dtype=np.float32)

    @property
    def successes(self) -> npt.NDArray[np.bool_]:  # (episode_length,)
        return np.asarray(a=self.group["success"], dtype=np.bool_)

    @property
    def simulator_states(self) -> dict:
        states = self.group["env_states"]
        return {
            group_name: {
                name: np.asarray(a=entry)  # (episode_length + 1, entry_dimensionality)
                for name, entry in states[group_name].items()
            }
            for group_name in states
        }


def _check_metadata(*, metadata: dict) -> None:
    environment_info = metadata["env_info"]
    if environment_info["env_id"] != TASK_ID:
        raise ValueError(
            f"Wrong task. The demonstrations were recorded for "
            f"{environment_info['env_id']!r}, not {TASK_ID!r}."
        )

    recorded_control_mode = environment_info["env_kwargs"].get("control_mode")
    if recorded_control_mode != CONTROL_MODE:
        raise ValueError(
            f"The demonstrations were recorded under {recorded_control_mode!r}, but "
            f"the pipeline drives {TASK_ID} with {CONTROL_MODE!r}."
        )


def _check_recording(*, recording: _Recording) -> None:
    if not bool(recording.successes[-1]):
        raise ValueError(
            f"Demonstration {recording.episode_id} does not end in success, although "
            f"every recorded {TASK_ID} demonstration is supposed to."
        )

    if recording.actions.shape[1] != ACTION_DIMENSIONALITY:
        raise ValueError(
            f"The demonstration holds {recording.actions.shape[1]}-entry actions, but "
            f"{TASK_ID} under {CONTROL_MODE} takes {ACTION_DIMENSIONALITY} entries."
        )


def _observe(*, gym_environment: Env) -> npt.NDArray[np.float32]:  # (state_count,)
    raw = (
        gym_environment.unwrapped.get_obs()  # type: ignore[attr-defined]
        .cpu()
        .numpy()
        .reshape(1, -1)
        .astype(np.float32)
    )  # (1, raw_dimensionality)
    return convert_observations(raw_observation_batch=raw)[0]


def _restore_episode(*, gym_environment: Env, recording: _Recording) -> _Episode:
    actions = recording.actions  # (episode_length, action_dimensionality)

    simulator_states = recording.simulator_states

    gym_environment.reset(seed=recording.seed)

    states = []
    for step_index in range(len(actions)):
        gym_environment.unwrapped.set_state_dict(  # type: ignore[attr-defined]
            {
                group_name: {
                    name: entry[step_index][None] for name, entry in group.items()
                }
                for group_name, group in simulator_states.items()
            }
        )
        states.append(_observe(gym_environment=gym_environment))

    rewards = recording.rewards / _DENSE_REWARD_SCALE  # (episode_length,)
    rewards[-1] = np.nan

    return _Episode(
        states=np.stack(arrays=states).astype(np.float32),
        actions=actions,
        rewards=rewards.astype(np.float32),
    )


def _check_against_replay(
    *, gym_environment: Env, recording: _Recording, episode: _Episode
) -> None:
    actions = recording.actions

    state, _ = gym_environment.reset(seed=recording.seed)
    replayed_states = [state]
    replayed_rewards = []
    for action in actions[:-1]:
        state, step_reward, _, _, _ = gym_environment.step(action)
        replayed_states.append(state)
        replayed_rewards.append(float(step_reward))

    state_deviation = float(
        np.abs(np.stack(arrays=replayed_states) - episode.states).max()
    )
    reward_deviation = float(
        np.abs(np.asarray(a=replayed_rewards) - episode.rewards[:-1]).max()
    )

    if max(state_deviation, reward_deviation) > _REPLAY_TOLERANCE:
        raise RuntimeError(
            f"Demonstration {recording.episode_id} restored from recorded simulator "
            f"states disagrees with replaying its actions: states differ by "
            f"{state_deviation:.3e}, rewards by {reward_deviation:.3e}. Either the "
            f"environment no longer matches the recording, or ManiSkill changed how "
            f"it normalizes dense rewards."
        )
