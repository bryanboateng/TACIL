from dataclasses import dataclass, field

import gym_pusht  # noqa: F401
import gymnasium
import numpy as np
import torch
from gymnasium import Env
from tqdm import tqdm

from scripts.evaluation.core import disturbance
from scripts.evaluation.core.condition import (
    Condition,
    DisturbanceLevel,
    FloorCondition,
)
from scripts.evaluation.core.renderer import RecoloredRenderWrapper, RenderSettings
from scripts.shared.policy_types import Policy


@dataclass(frozen=True)
class EpisodeRecord:
    seed: int
    reward: float
    terminated: bool
    steps: int
    final_observation: np.ndarray
    frames: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)


@dataclass
class _EpisodeRun:
    environment: Env
    seed: int
    observation: np.ndarray
    frames: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    best_reward: float = 0.0
    step_count: int = 0
    terminated: bool = False

    @property
    def record(self) -> EpisodeRecord:
        return EpisodeRecord(
            seed=self.seed,
            reward=self.best_reward,
            terminated=self.terminated,
            steps=self.step_count,
            final_observation=self.observation,
            frames=self.frames,
            rewards=self.rewards,
            actions=self.actions,
        )


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

    if condition.disturbance_level is not None:
        environment = disturbance.DisturbanceWrapper(
            environment=environment,
            magnitude=_get_disturbance_magnitude(
                floor_condition=condition.floor_condition,
                disturbance_level=condition.disturbance_level,
            ),
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
def roll_out_episodes(
    *, environments: list[Env], seeds: list[int], policy: Policy, render: bool
) -> list[EpisodeRecord]:
    episode_runs: list[_EpisodeRun] = []
    for environment, seed in zip(environments, seeds):
        observation, _ = environment.reset(seed=seed)
        environment.observation_space.seed(seed=seed)
        environment.action_space.seed(seed=seed)
        episode_runs.append(
            _EpisodeRun(
                environment=environment,
                seed=seed,
                observation=observation,
                frames=[_render_frame(environment=environment)] if render else [],
            )
        )

    running_episode_indices = list(range(len(episode_runs)))
    with tqdm(desc="Rollout", unit="step", leave=False) as progress:
        while len(running_episode_indices) > 0:
            progress.update()
            progress.set_postfix(running=len(running_episode_indices), refresh=False)
            actions_by_episode = policy(
                observations={
                    episode_index: episode_runs[episode_index].observation
                    for episode_index in running_episode_indices
                }
            )

            still_running_episode_indices: list[int] = []
            for episode_index in running_episode_indices:
                episode_run = episode_runs[episode_index]
                action = actions_by_episode[episode_index]

                observation, step_reward, terminated, truncated, _ = (
                    episode_run.environment.step(action=action)
                )
                episode_run.observation = observation
                episode_run.actions.append(action)
                episode_run.rewards.append(float(step_reward))
                episode_run.best_reward = max(
                    episode_run.best_reward, float(step_reward)
                )
                episode_run.step_count += 1

                if render:
                    episode_run.frames.append(
                        _render_frame(environment=episode_run.environment)
                    )

                if terminated or truncated:
                    episode_run.terminated = terminated
                else:
                    still_running_episode_indices.append(episode_index)

            running_episode_indices = still_running_episode_indices

    return [episode_run.record for episode_run in episode_runs]


# ---- Private --------------------------------------------


def _get_disturbance_magnitude(
    *, floor_condition: FloorCondition, disturbance_level: DisturbanceLevel
) -> float:
    match floor_condition:
        case FloorCondition.GRIPPY:
            match disturbance_level:
                case DisturbanceLevel.LOW:
                    return 1_500
                case DisturbanceLevel.HIGH:
                    return 4_500
        case FloorCondition.SLIPPERY:
            match disturbance_level:
                case DisturbanceLevel.LOW:
                    return 24
                case DisturbanceLevel.HIGH:
                    return 72


def _render_frame(environment: Env) -> np.ndarray:
    rendered = environment.render()
    if not isinstance(rendered, np.ndarray):
        raise TypeError(
            f"Expected rgb_array render to return ndarray, got {type(rendered)}"
        )
    return rendered
