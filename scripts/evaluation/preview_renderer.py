import itertools
import random
from pathlib import Path

import numpy as np
import pygame
import torch
import tyro
import wandb
from numpy import typing as npt

from scripts.data.base.episode_store import read_episode_store
from scripts.evaluation.core.condition import (
    Condition,
    DisturbanceLevel,
    FloorCondition,
)
from scripts.evaluation.core.renderer import RenderSettings
from scripts.evaluation.core.rollout import create_environment, roll_out_episodes
from scripts.evaluation.core.video import FREEZE_SECONDS, build_annotated_surfaces
from scripts.policies.gpi import config as gpi
from scripts.policies.gpi.policy import load_geometric_policy
from scripts.shared.log import create_logger, log_device_info
from scripts.shared.policy_types import Policy, PolicyFactory

FPS = 30
# The live preview renders small and without supersampling: smoothscaling the
# supersampled canvas is the dominant per-frame cost, and the on-screen window does not
# need the recording's resolution. Recorded videos keep the renderer's full-quality
# defaults.
RENDER_SIZE = 600


logger = create_logger()


def _build_gpi_policy_factory() -> PolicyFactory:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_device_info(device=device, logger=logger)

    config = gpi.Config(
        dataset_artifact="abc/expert-dataset:latest",
        noise=gpi.ObservationNoiseConfig(),
        retrieval=gpi.RetrievalConfig(),
        guidance=gpi.GuidanceConfig(),
        seed=42,
        action_horizon=8,
    )
    return load_geometric_policy(
        episode_store=read_episode_store(
            episode_file_path=(
                Path(wandb.Api().artifact(name=config.dataset_artifact).download())
            )
        ),
        config=config,
        device=device,
    )


def _build_avoidance_policy_factory() -> PolicyFactory:
    def factory(*, seeds: list[int]) -> Policy:
        del seeds

        def policy(
            *, observations: dict[int, npt.NDArray[np.float32]]
        ) -> dict[int, npt.NDArray[np.float32]]:
            actions = {}
            for episode_index, observation in observations.items():
                agent_position = observation[:2]
                block_position = observation[2:4]
                away_vector = agent_position - block_position
                distance = float(np.linalg.norm(away_vector))
                away_direction = (
                    away_vector / distance if distance > 1e-6 else np.array([1.0, 0.0])
                )
                target = block_position + away_direction * 150.0
                actions[episode_index] = target.astype(np.float32)
            return actions

        return policy

    return factory


def main(use_gpi_policy: bool = False) -> None:
    pygame.init()
    screen = pygame.display.set_mode(size=(RENDER_SIZE, RENDER_SIZE))
    clock = pygame.time.Clock()

    policy_factory = (
        _build_gpi_policy_factory()
        if use_gpi_policy
        else _build_avoidance_policy_factory()
    )

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
                policy=policy_factory(seeds=[seed]),
                render=True,
            )
        finally:
            environment.close()

        surfaces = build_annotated_surfaces(
            episode_record=episode_record,
            condition=condition,
            policy_label="dp (DDIM)   diffusion-policy-eternal-sweep-6:v0",
            notes="eval-lip-1-action-0.1-error-0.5",
            percentile=75,
        )

        # Play the episode, then hold on the final frame, matching the recorded freeze.
        freeze = [surfaces[-1]] * (FPS * FREEZE_SECONDS)
        for surface in [*surfaces, *freeze]:
            if _quit_requested():
                pygame.quit()
                return
            screen.blit(surface, (0, 0))
            pygame.display.flip()
            clock.tick(FPS)


def _quit_requested() -> bool:
    return any(event.type == pygame.QUIT for event in pygame.event.get())


if __name__ == "__main__":
    tyro.cli(f=main)
