import itertools
import random

import numpy as np
import pygame

from scripts.evaluation.core.condition import build_conditions
from scripts.evaluation.core.renderer import RENDER_SIZE
from scripts.evaluation.core.rollout import create_environment, roll_out_episode
from scripts.evaluation.core.video import (
    FREEZE_SECONDS,
    annotated_surfaces,
    create_font,
)
from scripts.shared.policy_types import Policy

PREVIEW_FPS = 30


def _figure_eight_policy_factory() -> Policy:
    step = itertools.count()
    amplitude = 150.0

    def policy(*, observation: np.ndarray) -> np.ndarray:
        t = next(step) * (2 * np.pi / 120)
        x = RENDER_SIZE / 2 + amplitude * np.cos(t)
        y = RENDER_SIZE / 2 + amplitude * np.sin(t) * np.cos(t)
        return np.asarray([x, y], dtype=np.float32)

    return policy


def main():
    pygame.init()
    screen = pygame.display.set_mode(size=(RENDER_SIZE, RENDER_SIZE))
    font = create_font()
    clock = pygame.time.Clock()
    rng = random.Random()

    for condition in itertools.cycle(build_conditions()):
        seed = rng.randint(a=0, b=2**31 - 1)
        pygame.display.set_caption(title="Preview")

        environment = create_environment(condition=condition, should_render=True)
        try:
            result = roll_out_episode(
                environment=environment,
                policy=_figure_eight_policy_factory(),
                seed=seed,
                render=True,
            )
        finally:
            environment.close()

        surfaces = annotated_surfaces(
            result=result,
            font=font,
            condition_label=str(condition),
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
