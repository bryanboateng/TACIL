from dataclasses import dataclass

import gym_pusht  # noqa: F401
import gymnasium
import numpy as np
import torch
from tqdm import tqdm

from scripts.evaluation.bryan.disturbance import DisturberConfig, setup_disturber
from scripts.shared.policy_types import PolicyFactory


@dataclass(frozen=True)
class RewardStatistics:
    mean: float
    std: float


@torch.no_grad()
def evaluate_policy(
    policy_factory: PolicyFactory,
    episode_seeds: tuple[int, ...],
    disturber_config: DisturberConfig | None,
    description: str,
) -> RewardStatistics:
    environment = gymnasium.make(id="gym_pusht/PushT-v0")

    if disturber_config is not None:
        environment.reset()
        setup_disturber(environment=environment, config=disturber_config)

    try:
        per_episode_reward: list[float] = []

        for episode_seed in tqdm(
            iterable=episode_seeds, desc=description, unit="episode", leave=False
        ):
            policy = policy_factory(seed=episode_seed)

            observation, _ = environment.reset(seed=episode_seed)
            environment.observation_space.seed(seed=episode_seed)
            environment.action_space.seed(seed=episode_seed)

            if disturber_config is not None:
                environment.unwrapped.disturber_rng = np.random.default_rng(  # ty:ignore[unresolved-attribute]
                    seed=episode_seed
                )

            episode_reward = 0.0
            while True:
                action = policy(observation=observation)
                observation, step_reward, terminated, truncated, _ = environment.step(
                    action=action
                )
                episode_reward = max(episode_reward, float(step_reward))
                if terminated or truncated:
                    break

            per_episode_reward.append(episode_reward)
    finally:
        environment.close()

    per_episode_reward_np = np.array(per_episode_reward)
    return RewardStatistics(
        mean=float(per_episode_reward_np.mean()), std=float(per_episode_reward_np.std())
    )
