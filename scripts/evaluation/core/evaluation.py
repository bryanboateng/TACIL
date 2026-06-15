from dataclasses import dataclass

import gym_pusht  # noqa: F401
import gymnasium
import numpy as np
import torch
from tqdm import tqdm

from scripts.evaluation.core.condition import EvalCondition
from scripts.evaluation.core.disturbance import setup_disturber
from scripts.shared.policy_types import PolicyFactory


@dataclass(frozen=True)
class RewardStatistics:
    mean: float
    std: float


@torch.no_grad()
def evaluate_policy(
    *,
    policy_factory: PolicyFactory,
    start_episode_seed: int,
    episode_count: int,
    condition: EvalCondition,
) -> RewardStatistics:
    environment = gymnasium.make(id="gym_pusht/PushT-v0", damping=condition.damping)

    if condition.disturber is not None:
        environment.reset()
        setup_disturber(environment=environment, config=condition.disturber)

    try:
        per_episode_reward: list[float] = []

        for episode_seed in tqdm(
            iterable=range(start_episode_seed, start_episode_seed + episode_count),
            desc=condition.name,
            unit="episode",
            leave=False,
        ):
            policy = policy_factory(seed=episode_seed)

            observation, _ = environment.reset(seed=episode_seed)
            environment.observation_space.seed(seed=episode_seed)
            environment.action_space.seed(seed=episode_seed)

            if condition.disturber is not None:
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
