from dataclasses import dataclass
from typing import Callable

import gym_pusht  # noqa: F401
import gymnasium
import numpy as np
import torch
from numpy import typing as npt
from tqdm import tqdm

from scripts.evaluation.bryan.disturbance import DisturberConfig, setup_disturber

# A policy maps an observation to an action; a factory builds one per episode seed
# (the seed pins the policy's own sampling, e.g. diffusion noise).
Policy = Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]]
PolicyFactory = Callable[[int], Policy]


@dataclass(frozen=True)
class MaxRewardStatistics:
    mean: float
    std: float


@torch.no_grad()
def evaluate_policy(
    policy_factory: PolicyFactory,
    episode_seeds: tuple[int, ...],
    disturber_config: DisturberConfig | None,
) -> MaxRewardStatistics:
    environment = gymnasium.make(id="gym_pusht/PushT-v0")

    if disturber_config is not None:
        # Reset once before patching so block/agent/space are populated.
        environment.reset()
        setup_disturber(environment=environment, config=disturber_config)

    try:
        per_episode_max_reward: list[float] = []

        for episode_seed in tqdm(
            episode_seeds, desc="Evaluating", leave=False, position=1
        ):
            policy = policy_factory(episode_seed)

            observation, _ = environment.reset(seed=episode_seed)
            environment.observation_space.seed(seed=episode_seed)
            environment.action_space.seed(seed=episode_seed)

            if disturber_config is not None:
                environment.unwrapped.disturber_rng = np.random.default_rng(  # ty:ignore[unresolved-attribute]
                    seed=episode_seed
                )

            maximum_reward = 0.0
            while True:
                action = policy(observation)
                observation, reward, terminated, truncated, _ = environment.step(
                    action=action
                )
                maximum_reward = max(maximum_reward, float(reward))
                if terminated or truncated:
                    break

            per_episode_max_reward.append(maximum_reward)
    finally:
        environment.close()

    max_reward_array = np.array(per_episode_max_reward)
    return MaxRewardStatistics(
        mean=float(max_reward_array.mean()), std=float(max_reward_array.std())
    )
