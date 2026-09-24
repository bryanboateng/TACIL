"""Guard rails for PegInsertionSide resets.

An earlier attempt to speed resets up kept the robot and table alive across
episodes and rebuilt only the peg and hole. That left PhysX state behind and
made a seed's outcome depend on how many episodes had run before it in the same
environment, which the test below would have caught. The scene is built by
unmodified ManiSkill code again; the only change is that the process holds one
render material so SAPIEN does not tear its renderer down between resets.
"""

import unittest
from contextlib import ExitStack

import numpy as np
import torch

from scripts.environments.peg_insertion.environment import (
    create_base_environment,
    keep_sapien_renderer_alive,
)
from scripts.environments.peg_insertion.observation import (
    extract_state_from_environment,
)
from scripts.environments.render_settings import RenderSettings

CONTACT_HEAVY_SEED = 108
PRIOR_SEEDS = (100, 104)
EPISODE_STEP_COUNT = 450


def run_episode(environment, seed):
    environment.reset(seed=seed)
    generator = np.random.default_rng(seed)
    for _ in range(EPISODE_STEP_COUNT):
        environment.step(generator.uniform(-0.3, 0.3, 7).astype(np.float32))
    return np.asarray(
        extract_state_from_environment(peg_insertion_side_env=environment.unwrapped)
    )


class PegSceneDeterminismTests(unittest.TestCase):
    def test_earlier_episodes_do_not_change_a_later_seed(self):
        thread_count = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with ExitStack() as resources:
                fresh = create_base_environment(episode_step_limit=EPISODE_STEP_COUNT)
                resources.callback(fresh.close)
                used = create_base_environment(episode_step_limit=EPISODE_STEP_COUNT)
                resources.callback(used.close)

                for seed in PRIOR_SEEDS:
                    run_episode(used, seed)

                np.testing.assert_array_equal(
                    run_episode(used, CONTACT_HEAVY_SEED),
                    run_episode(fresh, CONTACT_HEAVY_SEED),
                )
        finally:
            torch.set_num_threads(thread_count)

    def test_the_renderer_reference_is_held_for_the_whole_process(self):
        self.assertIs(keep_sapien_renderer_alive(), keep_sapien_renderer_alive())

    def test_a_rendering_environment_still_starts_after_a_headless_one(self):
        with ExitStack() as resources:
            headless = create_base_environment(episode_step_limit=EPISODE_STEP_COUNT)
            resources.callback(headless.close)
            headless.reset(seed=100)
            rendered = create_base_environment(
                episode_step_limit=EPISODE_STEP_COUNT,
                render_settings=RenderSettings(size=256, supersample_factor=1),
            )
            resources.callback(rendered.close)
            rendered.reset(seed=100)
            self.assertIsInstance(rendered.render(), np.ndarray)
