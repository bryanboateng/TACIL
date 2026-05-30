import gym_pusht  # noqa: F401
import gymnasium
import numpy as np
import numpy.typing as npt
import tyro
import zarr
from tqdm import tqdm

from scripts.ashared import create_logger, set_seed
from scripts.shared.paths import CCIL_ROOT

logger = create_logger()


def main(
    seed: int = 42,
    episode_count: int = 100,
    max_steps_per_episode: int = 200,
    output_file_name: str = "random_rollouts.zarr",
) -> None:
    set_seed(seed=seed)

    output_file_path = CCIL_ROOT / output_file_name
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Output: %s", output_file_path)

    environment = gymnasium.make("gym_pusht/PushT-v0")
    seeding_rng = np.random.default_rng(seed=seed)

    episode_state_trajectories: list[npt.NDArray[np.float32]] = []
    episode_action_trajectories: list[npt.NDArray[np.float32]] = []
    episode_lengths: list[int] = []

    for _ in tqdm(range(episode_count)):
        episode_seed = int(seeding_rng.integers(low=0, high=2**31 - 1))
        reset_observation, _ = environment.reset(seed=episode_seed)

        state_trajectory: list[npt.NDArray[np.float32]] = [
            np.asarray(reset_observation, dtype=np.float32)
        ]
        action_trajectory: list[npt.NDArray[np.float32]] = []

        for _ in range(max_steps_per_episode):
            action = np.asarray(environment.action_space.sample(), dtype=np.float32)
            next_observation, _, terminated, truncated, _ = environment.step(
                action=action
            )
            state_trajectory.append(np.asarray(next_observation, dtype=np.float32))
            action_trajectory.append(action)
            if terminated or truncated:
                break

        if len(action_trajectory) == 0:
            continue

        # Pad the action trajectory so it has the same length as the state trajectory.
        action_trajectory.append(np.zeros_like(action_trajectory[0]))

        episode_state_trajectories.append(np.stack(arrays=state_trajectory, axis=0))
        episode_action_trajectories.append(np.stack(arrays=action_trajectory, axis=0))
        episode_lengths.append(len(state_trajectory))

    environment.close()

    flat_states = np.concatenate(episode_state_trajectories, axis=0)
    flat_actions = np.concatenate(episode_action_trajectories, axis=0)
    episode_ends = np.cumsum(np.asarray(episode_lengths, dtype=np.int64))

    root = zarr.open_group(store=str(output_file_path), mode="w")
    data_group = root.require_group(name="data")
    meta_group = root.require_group(name="meta")
    data_group.create_array(name="state", data=flat_states)
    data_group.create_array(name="action", data=flat_actions)
    meta_group.create_array(name="episode_ends", data=episode_ends)

    transition_count = int(flat_states.shape[0] - len(episode_lengths))
    logger.info(
        "Saved %d transitions across %d episodes to %s",
        transition_count,
        len(episode_lengths),
        output_file_path,
    )


if __name__ == "__main__":
    tyro.cli(main)
