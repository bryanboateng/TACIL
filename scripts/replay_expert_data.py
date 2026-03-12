from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium
import numpy as np
import zarr


def main():
    states, actions, episode_ends = load_dataset()

    environment = gymnasium.make("gym_pusht/PushT-v0", render_mode="human")

    episode_start = 0

    for episode_end in episode_ends:
        set_environment_state(environment=environment, state=states[episode_start])

        for step in range(episode_start, episode_end - 1):
            observation, reward, terminated, truncated, info = environment.step(
                action=actions[step]
            )
            environment.render()

            print(f"Distance: {np.linalg.norm(observation - states[step + 1]):.4f}")

            if terminated or truncated:
                break

        episode_start = episode_end

    environment.close()


def load_dataset():
    dataset = zarr.open(
        store=Path("../expert_data/pusht_cchi_v7_replay.zarr"), mode="r"
    )
    states = dataset["data/state"][:]
    actions = dataset["data/action"][:]
    episode_ends = dataset["meta/episode_ends"][:]
    return states, actions, episode_ends


def set_environment_state(environment, state):
    environment.reset()

    unwrapped_environment = environment.unwrapped
    unwrapped_environment._set_state(state)

    unwrapped_environment.agent.position = tuple(state[0:2])
    unwrapped_environment.block.position = tuple(state[2:4])
    unwrapped_environment.block.angle = float(state[4])


if __name__ == "__main__":
    main()
