import itertools
import random
from pathlib import Path

import pygame
import torch
import tyro
import wandb
from rich.pretty import pretty_repr

from scripts.evaluation.core.condition import (
    Condition,
    DisturbanceLevel,
    FloorCondition,
)
from scripts.evaluation.core.renderer import RenderSettings
from scripts.evaluation.core.rollout import create_environment, roll_out_episodes
from scripts.evaluation.core.video import (
    FREEZE_SECONDS,
    get_video_fps,
    render_episode_surfaces,
)
from scripts.policies.gpi import config as gpi
from scripts.policies.gpi.policy import create_geometric_policy_factory
from scripts.shared.episode_store import read_episode_store
from scripts.shared.log import create_logger, log_device_info
from scripts.shared.policy_types import PolicyFactory

RENDER_SIZE = 600


logger = create_logger()


def main() -> None:
    config = tyro.cli(f=gpi.Config)
    logger.info(pretty_repr(config))
    _run(config=config)


def _run(*, config: gpi.Config) -> None:
    pygame.init()
    screen = pygame.display.set_mode(size=(RENDER_SIZE, RENDER_SIZE))
    clock = pygame.time.Clock()

    fps = get_video_fps()
    policy_factory = _create_policy_factory(config=config)

    conditions = [
        Condition(floor_condition=floor_condition, disturbance_level=disturbance_level)
        for floor_condition in FloorCondition
        for disturbance_level in (None, DisturbanceLevel.LOW, DisturbanceLevel.HIGH)
    ]

    for condition in itertools.cycle(conditions):
        seed = random.randint(a=0, b=2**31 - 1)
        pygame.display.set_caption(title="Preview")

        environment = create_environment(
            condition=condition,
            render_settings=RenderSettings(size=RENDER_SIZE, supersample_factor=1),
        )
        try:
            (episode_record,) = roll_out_episodes(
                environments=[environment],
                seeds=[seed],
                policy=policy_factory(episode_seeds=[seed]),
                render=True,
            )
        finally:
            environment.close()

        surfaces = render_episode_surfaces(
            episode_record=episode_record,
            condition=condition,
            policy_label="dp (eternal-sweep-6)",
            percentile=75,
        )

        # Play the episode, then hold on the final frame, matching the recorded freeze.
        freeze = [surfaces[-1]] * (fps * FREEZE_SECONDS)
        for surface in [*surfaces, *freeze]:
            if _quit_requested():
                pygame.quit()
                return
            screen.blit(surface, (0, 0))
            pygame.display.flip()
            clock.tick(fps)


def _create_policy_factory(config: gpi.Config) -> PolicyFactory:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_device_info(device=device, logger=logger)

    dataset_artifact = "abc/expert-dataset:latest"
    return create_geometric_policy_factory(
        episode_store=read_episode_store(
            episode_file_path=(
                Path(wandb.Api().artifact(name=dataset_artifact).download())
            )
        ),
        config=config,
        device=device,
    )


def _quit_requested() -> bool:
    return any(event.type == pygame.QUIT for event in pygame.event.get())


if __name__ == "__main__":
    tyro.cli(f=main)
