import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import mani_skill.envs  # noqa: F401
import numpy as np
from gymnasium import Env
from mani_skill import PegInsertionSideEnv
from mani_skill.trajectory.replay_trajectory import Args as ReplayArgs
from mani_skill.trajectory.replay_trajectory import main as replay_trajectory
from mani_skill.utils.download_demo import DATASET_SOURCES
from numpy import typing as npt
from tqdm import tqdm

from scripts.environments.peg_insertion.environment import (
    CONTROL_MODE,
    REWARD_MODE,
    TASK_ID,
    create_base_environment,
)
from scripts.environments.peg_insertion.observation import (
    extract_state_from_environment,
)
from scripts.shared.download import download_and_extract_zip
from scripts.shared.episode_store import RewardedEpisodeStore
from scripts.shared.log import create_logger

logger = create_logger()

SimulatorStates = dict[str, dict[str, npt.NDArray[Any]]]


def build_episode_store() -> RewardedEpisodeStore:
    dataset_path = _get_dataset(
        dataset_directory_path=(
            download_and_extract_zip(url=DATASET_SOURCES[TASK_ID].raw_dataset_url)
            / TASK_ID
            / "motionplanning"
        )
    )

    metadata = json.loads(s=(dataset_path.with_suffix(suffix=".json")).read_text())
    _check_metadata(metadata=metadata)

    episodes = _build_episodes(
        dataset_path=dataset_path, episode_metadata=metadata["episodes"]
    )

    return RewardedEpisodeStore(
        states=np.concatenate(
            [episode.states for episode in episodes]
        ),  # (total_timestep_count, STATE_DIMENSIONALITY)
        actions=np.concatenate(
            [episode.actions for episode in episodes]
        ),  # (total_timestep_count, ACTION_DIMENSIONALITY)
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
class _Episode:
    states: npt.NDArray[np.float32]  # (episode_length, STATE_DIMENSIONALITY)
    actions: npt.NDArray[np.float32]  # (episode_length, ACTION_DIMENSIONALITY)
    rewards: npt.NDArray[np.float32]  # (episode_length,)


def _get_dataset(*, dataset_directory_path: Path) -> Path:
    logger.info("Retrieving dataset at %s.", dataset_directory_path)

    joint_control_dataset_path = dataset_directory_path / "trajectory.h5"

    converted_dataset_path = _find_converted_dataset(
        dataset_directory_path=dataset_directory_path
    )

    if converted_dataset_path is not None:
        logger.info("Using cached conversion at %s.", converted_dataset_path)
        return converted_dataset_path

    logger.info("Converting demonstrations to %s.", CONTROL_MODE)
    replay_trajectory(
        args=ReplayArgs(
            traj_path=str(joint_control_dataset_path),
            obs_mode="none",
            target_control_mode=CONTROL_MODE,
            save_traj=True,
            max_retry=2,
            record_rewards=True,
            reward_mode=REWARD_MODE,
        )
    )

    converted_dataset_path = _find_converted_dataset(
        dataset_directory_path=dataset_directory_path
    )
    if converted_dataset_path is None:
        raise RuntimeError("Trajectory conversion produced no new .h5 file.")

    logger.info("Conversion complete: %s.", converted_dataset_path)
    return converted_dataset_path


def _check_metadata(*, metadata: dict[str, Any]) -> None:
    environment_info = metadata["env_info"]
    environment_id = environment_info["env_id"]
    if environment_id != TASK_ID:
        raise ValueError(
            f"Wrong task. Expected {TASK_ID!r} but got {environment_id!r}."
        )

    env_kwargs = environment_info["env_kwargs"]

    control_mode = env_kwargs.get("control_mode")
    if control_mode != CONTROL_MODE:
        raise ValueError(
            f"Wrong control mode. Expected {CONTROL_MODE!r} but got {control_mode!r}."
        )

    reward_mode = env_kwargs.get("reward_mode")
    if reward_mode != REWARD_MODE:
        raise ValueError(
            f"Wrong reward mode. Expected {REWARD_MODE!r} but got {reward_mode!r}"
        )


def _find_converted_dataset(*, dataset_directory_path: Path) -> Path | None:
    for path in dataset_directory_path.glob("*.h5"):
        if path.name != "trajectory.h5":
            return path
    return None


def _build_episodes(
    *, dataset_path: Path, episode_metadata: list[dict]
) -> list[_Episode]:
    logger.info("Starting episode building")
    episodes: list[_Episode] = []

    with h5py.File(name=dataset_path, mode="r") as dataset_file:
        max_steps = max(
            len(dataset_file[f"traj_{metadata['episode_id']}"]["actions"])
            for metadata in episode_metadata
        )

        gym_environment = create_base_environment(max_episode_steps=max_steps + 1)
        try:
            for metadata in tqdm(
                iterable=episode_metadata, desc="Building episodes", unit="episode"
            ):
                episodes.append(
                    _build_episode(
                        gym_environment=gym_environment,
                        group=dataset_file[f"traj_{metadata['episode_id']}"],
                        seed=metadata["episode_seed"],
                    )
                )
        finally:
            gym_environment.close()

    logger.info("Completed episode building")
    return episodes


def _build_episode(*, gym_environment: Env, group: h5py.Group, seed: int) -> _Episode:
    actions = np.asarray(
        a=group["actions"], dtype=np.float32
    )  # (episode_length, ACTION_DIMENSIONALITY)
    rewards = np.asarray(a=group["rewards"], dtype=np.float32)  # (episode_length,)

    simulator_states = _get_simulator_states(group=group)

    gym_environment.reset(seed=seed)
    peg_insertion_side_env = gym_environment.unwrapped
    if not isinstance(peg_insertion_side_env, PegInsertionSideEnv):
        raise TypeError(
            f"Expected PegInsertionSideEnv, got {type(peg_insertion_side_env).__name__}"
        )

    states: list[npt.NDArray[np.float32]] = []
    for step_index in range(len(actions)):
        peg_insertion_side_env.set_state_dict(
            state={
                state_category_name: {
                    state_name: state_over_time[step_index][None]
                    for state_name, state_over_time in states_by_name.items()
                }
                for state_category_name, states_by_name in simulator_states.items()
            }
        )
        states.append(
            extract_state_from_environment(
                peg_insertion_side_env=peg_insertion_side_env
            )
        )

    return _Episode(
        states=np.stack(arrays=states).astype(np.float32),
        actions=actions,
        rewards=rewards,
    )


def _get_simulator_states(*, group: h5py.Group) -> SimulatorStates:
    env_states = group["env_states"]
    simulator_states = {
        state_category_name: {
            state_name: np.asarray(dataset)
            for state_name, dataset in state_group.items()
        }
        for state_category_name, state_group in env_states.items()
    }
    return simulator_states
