from dataclasses import dataclass, field

import gym_pusht  # noqa: F401
import gymnasium
import numpy as np
import torch
from gymnasium import Env

from scripts.evaluation.core import disturbance
from scripts.evaluation.core.condition import Condition, FloorCondition
from scripts.evaluation.core.renderer import RecoloredRenderWrapper, RenderSettings
from scripts.shared.policy_types import Policy


@dataclass(frozen=True)
class EpisodeResult:
    reward: float
    terminated: bool
    steps: int
    frames: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)


def create_environment(
    *, condition: Condition, render_settings: RenderSettings | None
) -> Env:
    match condition.floor_condition:
        case FloorCondition.GRIPPY:
            damping = 0.0
        case FloorCondition.SLIPPERY:
            damping = 0.2

    environment = gymnasium.make(
        id="gym_pusht/PushT-v0",
        damping=damping,
        render_mode="rgb_array" if render_settings is not None else None,
    )
    environment.reset()

    if condition.disturbance is not None:
        environment = disturbance.DisturbanceWrapper(
            environment=environment, magnitude=condition.disturbance.magnitude
        )

    if render_settings is not None:
        match condition.floor_condition:
            case FloorCondition.GRIPPY:
                colormap = "Blues"
            case FloorCondition.SLIPPERY:
                colormap = "Oranges"
        environment = RecoloredRenderWrapper(
            environment=environment, colormap=colormap, render_settings=render_settings
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


# ---- Private --------------------------------------------


def _render_frame(environment: Env) -> np.ndarray:
    rendered = environment.render()
    if not isinstance(rendered, np.ndarray):
        raise TypeError(
            f"Expected rgb_array render to return ndarray, got {type(rendered)}"
        )
    return rendered
