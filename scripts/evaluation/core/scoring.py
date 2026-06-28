from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

from scripts.evaluation.core.condition import EvaluationCondition
from scripts.evaluation.core.rollout import create_environment, roll_out_episode
from scripts.shared.policy_types import PolicyFactory

SUCCESS_THRESHOLD = 0.95


@dataclass(frozen=True)
class RewardStatistics:
    mean: float
    std: float


@dataclass(frozen=True)
class SelectedSeeds:
    failures: list[tuple[int, float]]
    successes: list[tuple[int, float]]


@torch.no_grad()
def run_scored_episodes(
    *,
    policy_factory: PolicyFactory,
    start_episode_seed: int,
    episode_count: int,
    condition: EvaluationCondition,
) -> list[tuple[int, float]]:
    """Score `episode_count` episodes under `condition`, returning (seed, reward) pairs."""
    environment = create_environment(condition=condition, render=False)
    try:
        seed_rewards: list[tuple[int, float]] = []
        for seed in tqdm(
            iterable=range(start_episode_seed, start_episode_seed + episode_count),
            desc=condition,
            unit="episode",
            leave=False,
        ):
            result = roll_out_episode(
                environment=environment,
                policy=policy_factory(seed=seed),
                seed=seed,
                render=False,
            )
            seed_rewards.append((seed, result.reward))
        return seed_rewards
    finally:
        environment.close()


def reward_statistics(seed_rewards: list[tuple[int, float]]) -> RewardStatistics:
    rewards = np.array([reward for _, reward in seed_rewards])
    return RewardStatistics(mean=float(rewards.mean()), std=float(rewards.std()))


def select_seeds(seed_rewards: list[tuple[int, float]]) -> SelectedSeeds:
    return SelectedSeeds(
        failures=_pick_representative(
            [sr for sr in seed_rewards if sr[1] < SUCCESS_THRESHOLD]
        ),
        successes=_pick_representative(
            [sr for sr in seed_rewards if sr[1] >= SUCCESS_THRESHOLD]
        ),
    )


def _pick_representative(
    seed_rewards: list[tuple[int, float]],
) -> list[tuple[int, float]]:
    """Pick up to three seeds spanning the reward range: worst, median, best."""
    sorted_seed_rewards = sorted(seed_rewards, key=lambda sr: sr[1])
    count = len(sorted_seed_rewards)
    if count <= 2:
        return sorted_seed_rewards
    return [
        sorted_seed_rewards[0],
        sorted_seed_rewards[count // 2],
        sorted_seed_rewards[-1],
    ]
