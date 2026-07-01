import itertools
import random
from pathlib import Path

import pygame
import torch
import wandb

from scripts.evaluation.core.disturbance.condition import build_conditions, label
from scripts.evaluation.core.rollout import create_environment, roll_out_episode
from scripts.evaluation.core.video import (
    FREEZE_SECONDS,
    build_annotated_surfaces,
    create_font,
)
from scripts.policies.gpi import config as gpi
from scripts.policies.gpi.policy import load_geometric_policy
from scripts.shared.data import load_episode_store
from scripts.shared.log import create_logger, log_device_info
from scripts.shared.policy_types import PolicyFactory

PREVIEW_FPS = 30
# The live preview renders small and without supersampling: smoothscaling the
# supersampled canvas is the dominant per-frame cost, and the on-screen window does not
# need the recording's resolution. Recorded videos keep the renderer's full-quality
# defaults.
PREVIEW_RENDER_SIZE = 600
PREVIEW_SUPERSAMPLE_FACTOR = 1


logger = create_logger()


def _build_gpi_policy_factory() -> PolicyFactory:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_device_info(device=device, logger=logger)

    config = gpi.Config(
        dataset_artifact="abc/expert-dataset:latest",
        noise=gpi.ObservationNoiseConfig(),
        retrieval=gpi.RetrievalConfig(),
        guidance=gpi.GuidanceConfig(),
        action_horizon=8,
    )
    return load_geometric_policy(
        episode_store=load_episode_store(
            episode_file_path=(
                Path(wandb.Api().artifact(name=config.dataset_artifact).download())
            )
        ),
        config=config,
        device=device,
    )


def main():
    pygame.init()
    screen = pygame.display.set_mode(size=(PREVIEW_RENDER_SIZE, PREVIEW_RENDER_SIZE))
    font = create_font(render_size=PREVIEW_RENDER_SIZE)
    clock = pygame.time.Clock()

    policy_factory = _build_gpi_policy_factory()

    for condition in itertools.cycle(build_conditions()):
        seed = random.randint(a=0, b=2**31 - 1)
        pygame.display.set_caption(title="Preview")

        environment = create_environment(
            condition=condition,
            should_render=True,
            render_size=PREVIEW_RENDER_SIZE,
            supersample_factor=PREVIEW_SUPERSAMPLE_FACTOR,
        )
        try:
            episode_result = roll_out_episode(
                environment=environment,
                policy=policy_factory(seed=seed),
                seed=seed,
                render=True,
            )
        finally:
            environment.close()

        surfaces = build_annotated_surfaces(
            episode_result=episode_result,
            font=font,
            condition_label=label(condition),
            seed=seed,
            reward=None,
        )

        # Play the episode, then hold on the final frame, matching the recorded freeze.
        freeze = [surfaces[-1]] * (PREVIEW_FPS * FREEZE_SECONDS)
        for surface in [*surfaces, *freeze]:
            if _quit_requested():
                pygame.quit()
                return
            screen.blit(surface, (0, 0))
            pygame.display.flip()
            clock.tick(PREVIEW_FPS)


def _quit_requested() -> bool:
    return any(event.type == pygame.QUIT for event in pygame.event.get())


if __name__ == "__main__":
    main()
