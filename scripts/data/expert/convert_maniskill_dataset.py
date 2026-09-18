import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import tyro
from gymnasium import Env
from numpy import typing as npt
from rich.pretty import pretty_repr
from tqdm import tqdm

from scripts.environments.peg_insertion.condition import UNDISTURBED_CONDITION
from scripts.environments.peg_insertion.environment import (
    CONTROL_MODE,
    MAX_EPISODE_STEPS,
    TASK_ID,
    PegInsertionEnvironment,
)
from scripts.environments.peg_insertion.observation import ACTION_DIMENSIONALITY
from scripts.shared.episode_store import EpisodeStore, write_episode_store
from scripts.shared.log import create_logger

logger = create_logger()


@dataclass
class Config:
    demonstration_directory_path: Path
    output_file_path: Path
    episode_count: int | None = None


def main() -> None:
    config = tyro.cli(f=Config)
    logger.info(pretty_repr(config))
    _run(config=config)


def _run(*, config: Config) -> None:
    episode_store = replay_maniskill_demonstrations(
        demonstration_directory_path=config.demonstration_directory_path,
        episode_count=config.episode_count,
    )

    write_episode_store(
        episode_store=episode_store, save_file_path=config.output_file_path
    )

    episode_lengths = np.diff(
        a=episode_store.episode_end_indices, prepend=0
    )  # (episode_count,)

    if episode_store.rewards is None:
        raise RuntimeError("The replayed episodes unexpectedly carry no rewards.")

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
    logger.info(
        "Reward: min %.3f, max %.3f, mean %.3f",
        episode_store.rewards.min(),
        episode_store.rewards.max(),
        episode_store.rewards.mean(),
    )


def replay_maniskill_demonstrations(
    *, demonstration_directory_path: Path, episode_count: int | None
) -> EpisodeStore:
    metadata = json.loads(
        s=(demonstration_directory_path / "trajectory.json").read_text()
    )
    _check_metadata(metadata=metadata)

    recorded_episodes = metadata["episodes"]
    if episode_count is not None:
        recorded_episodes = recorded_episodes[:episode_count]

    environment = PegInsertionEnvironment()
    gym_environment = environment.create_gym_environment(
        condition=UNDISTURBED_CONDITION, render_settings=None
    )

    state_arrays: list[npt.NDArray[np.float32]] = []
    action_arrays: list[npt.NDArray[np.float32]] = []
    reward_arrays: list[npt.NDArray[np.float32]] = []
    episode_end_indices: list[int] = []
    cumulative_transition_count = 0
    unsolved_count = 0

    try:
        with h5py.File(
            name=demonstration_directory_path / "trajectory.h5", mode="r"
        ) as trajectory_file:
            for recorded_episode in tqdm(
                iterable=recorded_episodes, desc="Replaying", unit="episode"
            ):
                replayed = _replay_episode(
                    gym_environment=gym_environment,
                    seed=recorded_episode["episode_seed"],
                    recorded_actions=np.asarray(
                        a=trajectory_file[f"traj_{recorded_episode['episode_id']}"][
                            "actions"
                        ],
                        dtype=np.float32,
                    ),
                )
                if replayed is None:
                    unsolved_count += 1
                    continue

                state_arrays.append(replayed.states)
                action_arrays.append(replayed.actions)
                reward_arrays.append(replayed.rewards)

                cumulative_transition_count += len(replayed.actions)
                episode_end_indices.append(cumulative_transition_count)
    finally:
        gym_environment.close()

    if len(episode_end_indices) == 0:
        raise ValueError(
            f"None of the demonstrations in {demonstration_directory_path} solved "
            f"{TASK_ID} within {MAX_EPISODE_STEPS} steps."
        )

    if unsolved_count > 0:
        logger.info(
            "Dropped %d of %d demonstrations that did not solve the task within "
            "%d steps.",
            unsolved_count,
            len(recorded_episodes),
            MAX_EPISODE_STEPS,
        )

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
        rewards=np.concatenate(reward_arrays),  # (total_timestep_count,)
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
    seed: int,
    recorded_actions: npt.NDArray[np.float32],  # (recorded_length, action_count)
) -> _ReplayedEpisode | None:
    if recorded_actions.shape[1] != ACTION_DIMENSIONALITY:
        raise ValueError(
            f"The demonstration holds {recorded_actions.shape[1]}-entry actions, but "
            f"{TASK_ID} under {CONTROL_MODE} takes {ACTION_DIMENSIONALITY} entries."
        )

    state, _ = gym_environment.reset(seed=seed)

    states: list[npt.NDArray[np.float32]] = []
    actions: list[npt.NDArray[np.float32]] = []
    rewards: list[float] = []
    solved = False

    for action in recorded_actions[: MAX_EPISODE_STEPS - 1]:
        states.append(state)
        actions.append(action)

        state, step_reward, terminated, truncated, _ = gym_environment.step(action)
        rewards.append(float(step_reward))

        if terminated:
            solved = True
            break
        if truncated:
            break

    if not solved:
        return None

    held_action = actions[-1]
    states.append(state)
    actions.append(held_action)
    rewards.append(float(gym_environment.step(held_action)[1]))

    return _ReplayedEpisode(
        states=np.stack(arrays=states).astype(np.float32),
        actions=np.stack(arrays=actions).astype(np.float32),
        rewards=np.asarray(a=rewards, dtype=np.float32),
    )


if __name__ == "__main__":
    main()
