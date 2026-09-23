"""Integration checks against ManiSkill's original scene reconstruction."""

import unittest
from contextlib import ExitStack

import numpy as np
import torch

from scripts.environments.peg_insertion.disturbance import DisturbanceWrapper
from scripts.environments.peg_insertion.environment import create_base_environment
from scripts.environments.peg_insertion.observation import (
    extract_state_from_environment,
)


class ReusablePegSceneTests(unittest.TestCase):
    def assert_state_equal(self, original, reused):
        if isinstance(original, dict):
            self.assertEqual(original.keys(), reused.keys())
            for key in original:
                self.assert_state_equal(original[key], reused[key])
        else:
            np.testing.assert_array_equal(np.asarray(original), np.asarray(reused))

    def test_seeded_rollouts_match_and_static_objects_are_reused(self):
        thread_count = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for magnitude in (None, 0.02, 0.04):
                with self.subTest(magnitude=magnitude), ExitStack() as resources:
                    original = create_base_environment(
                        max_episode_steps=450, reuse_scene=False
                    )
                    resources.callback(original.close)
                    reused = create_base_environment(max_episode_steps=450)
                    resources.callback(reused.close)
                    if magnitude is not None:
                        original = DisturbanceWrapper(
                            environment=original, magnitude=magnitude
                        )
                        reused = DisturbanceWrapper(
                            environment=reused, magnitude=magnitude
                        )
                    task = reused.unwrapped
                    robot = task.agent
                    table = task.table_scene
                    scene = task.scene
                    initial_actor_count = len(scene.actors)
                    geometries = []
                    for seed in (10000, 10001, 10000):
                        original.reset(seed=seed)
                        reused.reset(seed=seed)
                        self.assertIs(task.agent, robot)
                        self.assertIs(task.table_scene, table)
                        self.assertIs(task.scene, scene)
                        self.assertEqual(len(scene.actors), initial_actor_count)
                        geometries.append(task.peg_half_sizes.clone())
                        self.assert_state_equal(
                            original.unwrapped.get_state_dict(), task.get_state_dict()
                        )
                        rng = np.random.default_rng(seed)
                        for step in range(60):
                            action = rng.uniform(-0.25, 0.25, 7).astype(np.float32)
                            # Approach the table and close the gripper; include both
                            # disturbance steps (20 and 50), not just free motion.
                            action[2] = -0.5 if step < 30 else 0.1
                            action[6] = -1
                            _, reward_a, done_a, truncated_a, _ = original.step(action)
                            _, reward_b, done_b, truncated_b, _ = reused.step(action)
                            self.assertEqual(
                                (reward_a, done_a, truncated_a),
                                (reward_b, done_b, truncated_b),
                            )
                            self.assert_state_equal(
                                original.unwrapped.get_state_dict(),
                                task.get_state_dict(),
                            )
                            np.testing.assert_array_equal(
                                extract_state_from_environment(
                                    peg_insertion_side_env=original.unwrapped
                                ),
                                extract_state_from_environment(
                                    peg_insertion_side_env=task
                                ),
                            )
                    self.assertFalse(torch.equal(geometries[0], geometries[1]))
                    self.assertTrue(torch.equal(geometries[0], geometries[2]))
        finally:
            torch.set_num_threads(thread_count)


if __name__ == "__main__":
    unittest.main()
