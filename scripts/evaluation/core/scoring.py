from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

from scripts.evaluation.core.condition import Condition, DisturbanceLevel
from scripts.evaluation.core.rollout import create_environment, roll_out_episode
from scripts.shared.policy_types import PolicyFactory


@dataclass(frozen=True)
class EpisodeScore:
    seed: int
    reward: float


@dataclass(frozen=True)
class RewardStatistics:
    mean: float
    std: float


@torch.no_grad()
def run_scored_episodes(
    *,
    policy_factory: PolicyFactory,
    start_episode_seed: int,
    episode_count: int,
    condition: Condition,
) -> list[EpisodeScore]:
    environment = create_environment(condition=condition, render_settings=None)
    try:
        episode_scores: list[EpisodeScore] = []
        for seed in tqdm(
            iterable=range(start_episode_seed, start_episode_seed + episode_count),
            desc=condition.label,
            unit="episode",
            leave=False,
        ):
            result = roll_out_episode(
                environment=environment,
                policy=policy_factory(seed=seed),
                seed=seed,
                render=False,
            )
            episode_scores.append(EpisodeScore(seed=seed, reward=result.reward))
        return episode_scores
    finally:
        environment.close()


def reward_statistics(episode_scores: list[EpisodeScore]) -> RewardStatistics:
    rewards = np.array([score.reward for score in episode_scores])
    return RewardStatistics(mean=float(rewards.mean()), std=float(rewards.std()))


def pick_representative_scores(
    *, episode_scores: list[EpisodeScore]
) -> dict[int, EpisodeScore]:
    count = 8

    sorted_scores = sorted(episode_scores, key=lambda score: score.reward)
    last_index = len(sorted_scores) - 1
    representative_scores: dict[int, EpisodeScore] = {}
    for step in range(count):
        fraction = step / (count - 1)
        percentile = round(number=100 * fraction)
        representative_scores[percentile] = sorted_scores[
            round(number=last_index * fraction)
        ]
    return representative_scores
