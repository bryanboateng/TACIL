from dataclasses import dataclass
from typing import Callable

import gym_pusht  # noqa: F401
import gymnasium
import numpy as np
import torch
from numpy import typing as npt

from scripts.evaluation.bryan.disturbance import DisturberConfig, setup_disturber

# A policy maps an observation to an action; a factory builds one per episode seed
# (the seed pins the policy's own sampling, e.g. diffusion noise).
Policy = Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]]
PolicyFactory = Callable[[int], Policy]


@dataclass(frozen=True)
class EvaluationResult:
    max_coverage_mean: float
    max_coverage_std: float
    success_rate: float
    per_episode_max_coverage: tuple[float, ...]


@torch.no_grad()
def evaluate_policy(
    policy_factory: PolicyFactory,
    episode_seeds: tuple[int, ...],
    disturber_config: DisturberConfig | None,
) -> EvaluationResult:
    environment = gymnasium.make(id="gym_pusht/PushT-v0")

    if disturber_config is not None:
        # Reset once before patching so block/agent/space are populated.
        environment.reset()
        setup_disturber(environment=environment, config=disturber_config)

    try:
        per_episode_max_coverage: list[float] = []
        per_episode_success: list[bool] = []

        for episode_seed in episode_seeds:
            policy = policy_factory(episode_seed)

            observation, _ = environment.reset(seed=episode_seed)
            environment.observation_space.seed(seed=episode_seed)
            environment.action_space.seed(seed=episode_seed)

            if disturber_config is not None:
                environment.unwrapped.disturber_rng = np.random.default_rng(  # ty:ignore[unresolved-attribute]
                    seed=episode_seed
                )

            maximum_coverage = 0.0
            while True:
                action = policy(observation)
                observation, _, terminated, truncated, info = environment.step(
                    action=action
                )
                maximum_coverage = max(maximum_coverage, float(info["coverage"]))
                if terminated or truncated:
                    break

            per_episode_max_coverage.append(maximum_coverage)
            per_episode_success.append(maximum_coverage >= 0.95)
    finally:
        environment.close()

    coverage_array = np.array(per_episode_max_coverage)
    return EvaluationResult(
        max_coverage_mean=float(coverage_array.mean()),
        max_coverage_std=float(coverage_array.std()),
        success_rate=float(np.mean(per_episode_success)),
        per_episode_max_coverage=tuple(per_episode_max_coverage),
    )
