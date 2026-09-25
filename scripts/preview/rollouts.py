import itertools
import random
from dataclasses import dataclass
from pathlib import Path

import pygame
import torch
import tyro
import wandb
from rich.pretty import pretty_repr

from scripts.environments.environment import Environment
from scripts.environments.registry import EnvironmentName, create_environment
from scripts.environments.render_settings import RenderSettings
from scripts.evaluation.core.rollout import roll_out_episodes
from scripts.evaluation.core.video import FREEZE_SECONDS, render_episode_surfaces
from scripts.policies.gpi import config as gpi
from scripts.policies.gpi.policy import create_geometric_policy_factory
from scripts.shared.dataset_artifact import DATASET_PROJECT
from scripts.shared.episodes import drop_rewards, read_episodes
from scripts.shared.log import create_logger, log_device_info
from scripts.shared.policy_types import PolicyFactory

RENDER_SIZE = 600


logger = create_logger()


@dataclass
class Config:
    environment: EnvironmentName
    dataset_artifact: str
    policy: gpi.Config


def main() -> None:
    config = tyro.cli(f=Config)
    logger.info(pretty_repr(config))
    _run(config=config)


def _run(*, config: Config) -> None:
    pygame.init()
    screen = pygame.display.set_mode(size=(RENDER_SIZE, RENDER_SIZE))
    clock = pygame.time.Clock()

    environment = create_environment(name=config.environment)
    fps = environment.video_settings.video_frames_per_second
    policy_factory = _create_policy_factory(config=config, environment=environment)

    for condition in itertools.cycle(environment.conditions):
        seed = random.randint(a=0, b=2**31 - 1)
        pygame.display.set_caption(title="Preview")

        gym_environment = environment.create_gym_environment(
            condition=condition,
            render_settings=RenderSettings(size=RENDER_SIZE, supersample_factor=1),
        )
        try:
            with torch.no_grad():
                (episode_record,) = roll_out_episodes(
                    environments=[gym_environment],
                    seeds=[seed],
                    policy=policy_factory(episode_seeds=[seed]),
                    render=True,
                )
        finally:
            gym_environment.close()

        surfaces = render_episode_surfaces(
            episode_record=episode_record,
            condition_description=environment.describe_condition(condition=condition),
            video_settings=environment.video_settings,
            policy_label="preview",
            percentile=75,
        )

        freeze = [surfaces[-1]] * (fps * FREEZE_SECONDS)
        for surface in [*surfaces, *freeze]:
            if _quit_has_been_requested():
                pygame.quit()
                return
            screen.blit(surface, (0, 0))
            pygame.display.flip()
            clock.tick(fps)


def _create_policy_factory(
    *, config: Config, environment: Environment
) -> PolicyFactory:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_device_info(device=device, logger=logger)

    return create_geometric_policy_factory(
        gpi_geometry=environment.gpi_geometry,
        episodes=drop_rewards(
            episodes=read_episodes(
                directory_path=Path(
                    wandb.Api()
                    .artifact(name=f"{DATASET_PROJECT}/{config.dataset_artifact}")
                    .download()
                )
            )
        ),
        config=config.policy,
        device=device,
    )


def _quit_has_been_requested() -> bool:
    return any(event.type == pygame.QUIT for event in pygame.event.get())


if __name__ == "__main__":
    main()
