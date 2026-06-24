from dataclasses import dataclass, field

import gym_pusht  # noqa: F401
import gymnasium
import numpy as np
import torch
from gym_pusht.envs import PushTEnv
from gymnasium import Env

from scripts.evaluation.core.condition import EvalCondition
from scripts.evaluation.core.disturbance import setup_disturber
from scripts.evaluation.core.renderer import RENDER_SIZE, setup_renderer
from scripts.shared.policy_types import Policy


@dataclass(frozen=True)
class EpisodeResult:
    reward: float
    terminated: bool
    steps: int
    frames: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)


def make_environment(*, condition: EvalCondition, render: bool) -> Env:
    """Create a PushT environment configured for `condition`.

    The disturber and renderer patches must be applied after an initial reset so the
    agent/block/space exist, so this resets once before returning.
    """
    if render:
        environment = gymnasium.make(
            id="gym_pusht/PushT-v0",
            render_mode="rgb_array",
            visualization_width=RENDER_SIZE,
            visualization_height=RENDER_SIZE,
        )
    else:
        environment = gymnasium.make(id="gym_pusht/PushT-v0")

    environment.reset()

    push_t_environment = _get_push_t_environment(environment=environment)

    if condition.disturbance is not None:
        setup_disturber(
            environment=push_t_environment, disturbance=condition.disturbance
        )
    if render:
        setup_renderer(environment=push_t_environment, color_map=condition.bg_cmap)

    return environment


@torch.no_grad()
def roll_out_episode(
    *, environment: Env, policy: Policy, seed: int, render: bool
) -> EpisodeResult:
    """Run one episode to termination/truncation, collecting frames only when `render`."""
    observation, _ = environment.reset(seed=seed)
    environment.observation_space.seed(seed=seed)
    environment.action_space.seed(seed=seed)

    frames = [_render_frame(environment)] if render else []

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
            frames.append(_render_frame(environment))
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


def _get_push_t_environment(environment: Env) -> PushTEnv:
    unwrapped_environment = environment.unwrapped
    if not isinstance(unwrapped_environment, PushTEnv):
        raise TypeError(
            f"Expected PushTEnv, got {type(unwrapped_environment).__name__}"
        )
    return unwrapped_environment
