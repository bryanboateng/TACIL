from dataclasses import dataclass, field

import numpy as np
from gymnasium import Env
from tqdm import tqdm

from scripts.shared.policy_types import Policy


@dataclass(frozen=True)
class Episode:
    seed: int
    reward: float
    terminated: bool
    final_observation: np.ndarray
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    frames: list[np.ndarray] | None = None


@dataclass
class _EpisodeRun:
    environment: Env
    seed: int
    observation: np.ndarray
    frames: list[np.ndarray] | None = None
    rewards: list[float] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    terminated: bool = False

    @property
    def episode(self) -> Episode:
        return Episode(
            seed=self.seed,
            reward=max(self.rewards, default=float("nan")),
            terminated=self.terminated,
            final_observation=self.observation,
            actions=self.actions,
            rewards=self.rewards,
            frames=self.frames,
        )


def roll_out_episodes(
    *, environments: list[Env], seeds: list[int], policy: Policy, render: bool
) -> list[Episode]:
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
                frames=[_render_frame(environment=environment)] if render else None,
            )
        )

    running_episode_indices = list(range(len(episode_runs)))
    with tqdm(desc="Rollout", unit="step", leave=False, dynamic_ncols=True) as progress:
        while len(running_episode_indices) > 0:
            progress.update()
            progress.set_postfix(running=len(running_episode_indices), refresh=False)
            actions_by_episode_index = policy(
                observations_by_episode_index={
                    episode_index: episode_runs[episode_index].observation
                    for episode_index in running_episode_indices
                }
            )

            still_running_episode_indices: list[int] = []
            for episode_index in running_episode_indices:
                episode_run = episode_runs[episode_index]
                action = actions_by_episode_index[episode_index]

                observation, step_reward, terminated, truncated, _ = (
                    episode_run.environment.step(action=action)
                )
                episode_run.observation = observation
                episode_run.actions.append(action)
                episode_run.rewards.append(float(step_reward))

                if episode_run.frames is not None:
                    episode_run.frames.append(
                        _render_frame(environment=episode_run.environment)
                    )

                if terminated or truncated:
                    episode_run.terminated = terminated
                else:
                    still_running_episode_indices.append(episode_index)

            running_episode_indices = still_running_episode_indices

    return [episode_run.episode for episode_run in episode_runs]


# ---- Private --------------------------------------------


def _render_frame(environment: Env) -> np.ndarray:
    rendered = environment.render()
    if not isinstance(rendered, np.ndarray):
        raise TypeError(
            f"Expected rgb_array render to return ndarray, got {type(rendered)}"
        )
    return rendered
