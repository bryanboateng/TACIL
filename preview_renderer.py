"""Live renderer preview — cycles through all four eval conditions, one episode each."""

import itertools
import random

import gym_pusht  # noqa: F401
import gymnasium
import numpy as np
import pygame

from scripts.evaluation.core.disturbance import DisturberConfig, setup_disturber
from scripts.evaluation.core.renderer import setup_renderer

CONDITIONS = [
    # dict(name="baseline", bg_cmap="Greys", bg_val=0.2),
    dict(name="slidy", bg_cmap="Blues", bg_val=0.4),
    dict(name="disturbed", bg_cmap="Oranges", bg_val=0.4),
    dict(name="slidy+disturbed", bg_cmap="Reds", bg_val=0.4),
]


def main():
    rng = random.Random()

    env = gymnasium.make("gym_pusht/PushT-v0", render_mode="human")

    # Initial reset so agent/block/space exist before setup_disturber/setup_renderer
    env.reset(seed=0)
    setup_disturber(environment=env, config=DisturberConfig())

    for condition in itertools.cycle(CONDITIONS):
        setup_renderer(env, bg_cmap=condition["bg_cmap"])

        seed = rng.randint(0, 2**31 - 1)
        print(f"condition={condition['name']}  seed={seed}")
        env.reset(seed=seed)
        env.unwrapped.disturber_rng = np.random.default_rng(seed=seed)

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return

            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            env.render()

            if terminated or truncated:
                break


if __name__ == "__main__":
    main()
