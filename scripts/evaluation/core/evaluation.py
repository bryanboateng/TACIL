from dataclasses import dataclass

import gym_pusht  # noqa: F401
import gymnasium
import numpy as np
import pygame
import torch
import wandb
from tqdm import tqdm
from wandb import Run

from scripts.evaluation.core.condition import EvalCondition
from scripts.evaluation.core.disturbance import setup_disturber
from scripts.evaluation.core.renderer import setup_renderer
from scripts.shared.policy_types import PolicyFactory


@dataclass(frozen=True)
class RewardStatistics:
    mean: float
    std: float


@dataclass(frozen=True)
class SelectedSeeds:
    failures: list[tuple[int, float]]
    successes: list[tuple[int, float]]


@torch.no_grad()
def evaluate_policy(
    *,
    policy_factory: PolicyFactory,
    start_episode_seed: int,
    episode_count: int,
    condition: EvalCondition,
    run: Run,
) -> RewardStatistics:
    per_episode_reward = _run_episodes(
        policy_factory=policy_factory,
        start_episode_seed=start_episode_seed,
        episode_count=episode_count,
        condition=condition,
    )

    selected_seeds = _select_seeds(
        seed_rewards=list(
            zip(
                list(range(start_episode_seed, start_episode_seed + episode_count)),
                per_episode_reward,
            )
        )
    )
    _record(
        policy_factory=policy_factory,
        condition=condition,
        selected_seeds=selected_seeds,
        run=run,
    )

    rewards = np.array(per_episode_reward)
    return RewardStatistics(mean=float(rewards.mean()), std=float(rewards.std()))


@torch.no_grad()
def _run_episodes(
    *,
    policy_factory: PolicyFactory,
    start_episode_seed: int,
    episode_count: int,
    condition: EvalCondition,
) -> list[float]:
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

    return per_episode_reward


def pick_up_to_three(seed_rewards: list[tuple[int, float]]) -> list[tuple[int, float]]:
    sorted_seed_rewards = sorted(seed_rewards, key=lambda x: x[1])
    count = len(sorted_seed_rewards)
    if count == 0:
        return []
    if count <= 2:
        return sorted_seed_rewards
    return [
        sorted_seed_rewards[0],
        sorted_seed_rewards[count // 2],
        sorted_seed_rewards[-1],
    ]


def _select_seeds(seed_rewards: list[tuple[int, float]]) -> SelectedSeeds:
    return SelectedSeeds(
        failures=pick_up_to_three(
            seed_rewards=[
                seed_reward for seed_reward in seed_rewards if seed_reward[1] < 0.95
            ]
        ),
        successes=pick_up_to_three(
            seed_rewards=[
                seed_reward for seed_reward in seed_rewards if seed_reward[1] >= 0.95
            ]
        ),
    )


def _frame_to_surface(frame: np.ndarray) -> pygame.Surface:
    return pygame.surfarray.make_surface(frame.transpose(1, 0, 2))


def _surface_to_frame(surface: pygame.Surface) -> np.ndarray:
    return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))


def _annotate(
    surface: pygame.Surface,
    font: pygame.font.Font,
    seed: int,
    reward: float,
    badge: str | None,
) -> pygame.Surface:
    out = surface.copy()
    out.blit(
        source=font.render(
            text=f"seed {seed},  reward {reward:.2f}", antialias=True, color=(0, 0, 0)
        ),
        dest=(8, 8),
    )
    if badge is not None:
        badge_surf = font.render(text=badge, antialias=True, color=(0, 0, 0))
        out.blit(
            source=badge_surf,
            dest=(surface.get_width() - badge_surf.get_width() - 8, 8),
        )
    return out


@torch.no_grad()
def _record(
    *,
    policy_factory: PolicyFactory,
    condition: EvalCondition,
    selected_seeds: SelectedSeeds,
    run: Run,
) -> None:
    fps = 10
    freeze_duration = 2

    pygame.init()
    font = pygame.font.SysFont("Inter", 512 // 23)

    environment = gymnasium.make(
        id="gym_pusht/PushT-v0",
        render_mode="rgb_array",
        damping=condition.damping,
        visualization_width=512,
        visualization_height=512,
    )

    environment.reset()

    if condition.disturber is not None:
        setup_disturber(environment=environment, config=condition.disturber)

    setup_renderer(environment=environment, bg_color=condition.bg_color)

    try:
        for label, group in [
            ("failure", selected_seeds.failures),
            ("success", selected_seeds.successes),
        ]:
            if not group:
                continue

            all_frames: list[np.ndarray] = []

            for seed, reward in tqdm(
                group, desc=f"{condition.name}/{label}", unit="ep"
            ):
                policy = policy_factory(seed=seed)

                observation, _ = environment.reset(seed=seed)
                environment.observation_space.seed(seed=seed)
                environment.action_space.seed(seed=seed)

                if condition.disturber is not None:
                    environment.unwrapped.disturber_rng = np.random.default_rng(  # ty:ignore[unresolved-attribute]
                        seed=seed
                    )

                rendered = environment.render()
                if not isinstance(rendered, np.ndarray):
                    raise TypeError(
                        f"Expected rgb_array render to return ndarray, got {type(rendered)}"
                    )

                raw_frames = [rendered]

                step = 0
                succeeded = False
                while True:
                    action = policy(observation=observation)
                    observation, _, terminated, truncated, _ = environment.step(
                        action=action
                    )

                    rendered = environment.render()
                    if not isinstance(rendered, np.ndarray):
                        raise TypeError(
                            f"Expected rgb_array render to return ndarray, got {type(rendered)}"
                        )
                    raw_frames.append(rendered)

                    step += 1
                    if terminated:
                        succeeded = True
                        break
                    if truncated:
                        break

                badge = f"Success at step {step}" if succeeded else "Failure"

                for frame_index, frame in enumerate(raw_frames):
                    is_last = frame_index == len(raw_frames) - 1
                    surface = _annotate(
                        surface=_frame_to_surface(frame),
                        font=font,
                        seed=seed,
                        reward=reward,
                        badge=badge if is_last else None,
                    )
                    all_frames.append(_surface_to_frame(surface))

                # Freeze on the final annotated frame
                all_frames.extend([all_frames[-1]] * (fps * freeze_duration))

            # wandb.Video expects (T, C, H, W)
            run.summary[f"{condition.name}/video_{label}"] = wandb.Video(
                data_or_path=np.stack(all_frames).transpose(0, 3, 1, 2), fps=fps
            )
    finally:
        environment.close()
