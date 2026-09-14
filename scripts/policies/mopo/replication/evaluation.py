from dataclasses import dataclass

import gymnasium
import numpy as np
import torch

from scripts.policies.mopo.policy import MopoPolicy
from scripts.policies.mopo.replication.tasks import Task


@dataclass(frozen=True)
class EvaluationResult:
    score_mean: float
    score_std: float
    return_mean: float
    return_std: float
    episode_length_mean: float
    episode_count: int


def evaluate(
    *,
    policy: MopoPolicy,
    task: Task,
    episode_count: int,
    seed: int,
    device: torch.device,
) -> EvaluationResult:
    environment = gymnasium.make(task.environment_id)
    was_training = policy.actor.training
    policy.eval()

    returns: list[float] = []
    lengths: list[int] = []
    try:
        for episode_index in range(episode_count):
            state, _ = environment.reset(seed=seed + episode_index)
            episode_return = 0.0
            length = 0
            while True:
                with torch.no_grad():
                    action = (
                        policy.sample_action(
                            state_batch=torch.as_tensor(
                                data=state, dtype=torch.float32, device=device
                            ).unsqueeze(0),
                            deterministic=True,
                        )[0]
                        .squeeze(0)
                        .cpu()
                        .numpy()
                    )
                state, reward, terminated, truncated, _ = environment.step(action)
                episode_return += float(reward)
                length += 1
                if terminated or truncated:
                    break
            returns.append(episode_return)
            lengths.append(length)
    finally:
        environment.close()
        if was_training:
            policy.train()

    score_batch = np.array(
        [task.score(episode_return=one) for one in returns], dtype=np.float64
    )
    return EvaluationResult(
        score_mean=float(score_batch.mean()),
        score_std=float(score_batch.std()),
        return_mean=float(np.mean(returns)),
        return_std=float(np.std(returns)),
        episode_length_mean=float(np.mean(lengths)),
        episode_count=episode_count,
    )
