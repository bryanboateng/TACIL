import gym_pusht  # noqa: F401
import gymnasium


def main():
    environment = gymnasium.make(id="gym_pusht/PushT-v0", render_mode="human")
    observation, info = environment.reset()

    for _ in range(1000):
        action = environment.action_space.sample()
        observation, reward, terminated, truncated, info = environment.step(
            action=action
        )
        environment.render()

        if terminated or truncated:
            observation, info = environment.reset()

    environment.close()


if __name__ == "__main__":
    main()
