"""A reset must give the same result for a seed. The result must not change if
other episodes ran in the environment before.

We made a fault when we tried to make resets faster. We kept the robot and the
table. We rebuilt only the peg and the hole. Old physics data stayed in the
environment and changed the result.

We now rebuild the full scene at each reset. These tests check that a reused
environment and a new environment give the same state for the same seed."""

import unittest
from contextlib import ExitStack

import numpy as np
import torch

from scripts.environments.peg_insertion.environment import create_base_environment
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
