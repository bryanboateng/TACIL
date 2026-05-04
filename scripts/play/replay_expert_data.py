import gym_pusht  # noqa: F401
import gymnasium
import numpy as np

from scripts.shared import create_logger, load_push_t_zarr

logger = create_logger()


def main():
    states, actions, episode_ends = load_push_t_zarr()

    environment = gymnasium.make("gym_pusht/PushT-v0", render_mode="human")

    episode_start = 0

    for episode_end in episode_ends:
        set_environment_state(environment=environment, state=states[episode_start])

        for step in range(episode_start, episode_end - 1):
            observation, reward, terminated, truncated, info = environment.step(
                action=actions[step]
            )
            environment.render()

            logger.info(
                "Distance: %.4f", np.linalg.norm(observation - states[step + 1])
            )

            if terminated or truncated:
                break

        episode_start = episode_end

    environment.close()


def set_environment_state(environment, state):
    environment.reset()

    unwrapped_environment = environment.unwrapped

    # noinspection PyProtectedMember
    unwrapped_environment._set_state(state)  # noqa: SLF001 - there is no public API for setting state

    unwrapped_environment.agent.position = tuple(state[0:2])
    unwrapped_environment.block.position = tuple(state[2:4])
    unwrapped_environment.block.angle = float(state[4])


if __name__ == "__main__":
    main()
