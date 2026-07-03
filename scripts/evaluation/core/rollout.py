from dataclasses import dataclass, field

import gym_pusht  # noqa: F401
import gymnasium
import numpy as np
import torch
from gymnasium import Env

from scripts.evaluation.core.disturbance import block_push
from scripts.evaluation.core.disturbance.condition import Condition, condition_colormap
from scripts.evaluation.core.renderer import (
    RENDER_SIZE,
    SUPERSAMPLE_FACTOR,
    RecoloredRenderWrapper,
)
from scripts.shared.policy_types import Policy


@dataclass(frozen=True)
class EpisodeResult:
    reward: float
    terminated: bool
    steps: int
    frames: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)


def create_environment(
    *,
    condition: Condition,
    should_render: bool,
    render_size: int = RENDER_SIZE,
    supersample_factor: int = SUPERSAMPLE_FACTOR,
) -> Env:
    if should_render:
        environment: Env = gymnasium.make(
            id="gym_pusht/PushT-v0", render_mode="rgb_array", damping=condition.damping
        )
    else:
        environment = gymnasium.make(id="gym_pusht/PushT-v0", damping=condition.damping)

    environment.reset()

    if condition.push is not None:
        environment = block_push.BlockPushWrapper(
            environment=environment, config=condition.push
        )

    if should_render:
        environment = RecoloredRenderWrapper(
            environment=environment,
            colormap=condition_colormap(condition),
            render_size=render_size,
            supersample_factor=supersample_factor,
        )

    return environment


@torch.no_grad()
def roll_out_episode(
    *, environment: Env, policy: Policy, seed: int, render: bool
) -> EpisodeResult:
    observation, _ = environment.reset(seed=seed)
    environment.observation_space.seed(seed=seed)
    environment.action_space.seed(seed=seed)

    frames = [_render_frame(environment=environment)] if render else []

    reward = 0.0
    rewards: list[float] = []
    steps = 0
    while True:
        action = policy(observation=observation)
        observation, step_reward, terminated, truncated, _ = environment.step(
            action=action
        )
        rewards.append(float(step_reward))
        reward = max(reward, float(step_reward))
        steps += 1

        if render:
            frames.append(_render_frame(environment=environment))

        if terminated or truncated:
            return EpisodeResult(
                reward=reward,
                terminated=terminated,
                steps=steps,
                frames=frames,
                rewards=rewards,
            )


def _render_frame(environment: Env) -> np.ndarray:
    rendered = environment.render()
    if not isinstance(rendered, np.ndarray):
        raise TypeError(
            f"Expected rgb_array render to return ndarray, got {type(rendered)}"
        )
    return rendered
