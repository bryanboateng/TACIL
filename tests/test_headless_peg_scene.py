"""Integration checks against ManiSkill's original scene construction.

HeadlessPegInsertionSideEnv._load_scene mirrors PegInsertionSideEnv._load_scene,
omitting only the render materials and visual meshes that a headless run never
draws. Editing it must preserve the order of the episode RNG draws and every
collision shape, or seeded episodes silently stop matching upstream physics.

The history test below is the one that matters most: an earlier attempt kept the
robot and table across resets, which left PhysX state behind and made a seed's
outcome depend on how many episodes had run before it in the same environment.
"""

import unittest
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import sapien
import torch

from scripts.environments.peg_insertion.disturbance import DisturbanceWrapper
from scripts.environments.peg_insertion.environment import create_base_environment
from scripts.environments.peg_insertion.observation import (
    extract_state_from_environment,
)

CONTACT_HEAVY_SEED = 108
PRIOR_SEEDS = (100, 104)
EPISODE_STEP_COUNT = 450


def run_episode(environment, seed, step_count=EPISODE_STEP_COUNT):
    environment.reset(seed=seed)
    generator = np.random.default_rng(seed)
    for _ in range(step_count):
        environment.step(generator.uniform(-0.3, 0.3, 7).astype(np.float32))
    return np.asarray(
        extract_state_from_environment(peg_insertion_side_env=environment.unwrapped)
    )


class HeadlessPegSceneTests(unittest.TestCase):
    def test_earlier_episodes_do_not_change_a_later_seed(self):
        thread_count = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with ExitStack() as resources:
                fresh = create_base_environment(max_episode_steps=EPISODE_STEP_COUNT)
                resources.callback(fresh.close)
                used = create_base_environment(max_episode_steps=EPISODE_STEP_COUNT)
                resources.callback(used.close)
                upstream = create_base_environment(
                    max_episode_steps=EPISODE_STEP_COUNT, skip_unused_visuals=False
                )
                resources.callback(upstream.close)

                for seed in PRIOR_SEEDS:
                    run_episode(used, seed)

                expected = run_episode(upstream, CONTACT_HEAVY_SEED)
                np.testing.assert_array_equal(
                    run_episode(fresh, CONTACT_HEAVY_SEED), expected
                )
                np.testing.assert_array_equal(
                    run_episode(used, CONTACT_HEAVY_SEED), expected
                )
        finally:
            torch.set_num_threads(thread_count)

    def test_collision_geometry_matches_without_creating_render_materials(self):
        with ExitStack() as resources:
            upstream = create_base_environment(
                max_episode_steps=EPISODE_STEP_COUNT, skip_unused_visuals=False
            )
            resources.callback(upstream.close)
            headless = create_base_environment(max_episode_steps=EPISODE_STEP_COUNT)
            resources.callback(headless.close)
            for seed in (10000, 10001):
                upstream.reset(seed=seed)
                with patch(
                    "sapien.render.RenderMaterial",
                    side_effect=AssertionError(
                        "Headless reset created a render material"
                    ),
                ):
                    headless.reset(seed=seed)
                for name in ("peg", "box"):
                    bodies = [
                        getattr(environment.unwrapped, name)
                        ._objs[0]
                        .find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
                        for environment in (upstream, headless)
                    ]
                    self.assertEqual(bodies[0].mass, bodies[1].mass)
                    for a, b in zip(
                        bodies[0].get_collision_shapes(),
                        bodies[1].get_collision_shapes(),
                        strict=True,
                    ):
                        self.assertEqual(type(a), type(b))
                        np.testing.assert_array_equal(a.half_size, b.half_size)
                        np.testing.assert_array_equal(a.local_pose.p, b.local_pose.p)
                        np.testing.assert_array_equal(a.local_pose.q, b.local_pose.q)
                        for property_name in (
                            "static_friction",
                            "dynamic_friction",
                            "restitution",
                        ):
                            self.assertEqual(
                                getattr(a.physical_material, property_name),
                                getattr(b.physical_material, property_name),
                            )

    def assert_state_equal(self, upstream, headless):
        if isinstance(upstream, dict):
            self.assertEqual(upstream.keys(), headless.keys())
            for key in upstream:
                self.assert_state_equal(upstream[key], headless[key])
        else:
            np.testing.assert_array_equal(np.asarray(upstream), np.asarray(headless))

    def test_seeded_rollouts_match_upstream_under_every_condition(self):
        thread_count = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for magnitude in (None, 0.02, 0.04):
                with self.subTest(magnitude=magnitude), ExitStack() as resources:
                    upstream = create_base_environment(
                        max_episode_steps=EPISODE_STEP_COUNT, skip_unused_visuals=False
                    )
                    resources.callback(upstream.close)
                    headless = create_base_environment(
                        max_episode_steps=EPISODE_STEP_COUNT
                    )
                    resources.callback(headless.close)
                    if magnitude is not None:
                        upstream = DisturbanceWrapper(
                            environment=upstream, magnitude=magnitude
                        )
                        headless = DisturbanceWrapper(
                            environment=headless, magnitude=magnitude
                        )
                    geometries = []
                    for seed in (10000, 10001, 10000):
                        upstream.reset(seed=seed)
                        headless.reset(seed=seed)
                        geometries.append(headless.unwrapped.peg_half_sizes.clone())
                        self.assert_state_equal(
                            upstream.unwrapped.get_state_dict(),
                            headless.unwrapped.get_state_dict(),
                        )
                        generator = np.random.default_rng(seed)
                        for _ in range(120):
                            action = generator.uniform(-0.3, 0.3, 7).astype(np.float32)
                            _, reward_a, done_a, truncated_a, _ = upstream.step(action)
                            _, reward_b, done_b, truncated_b, _ = headless.step(action)
                            self.assertEqual(
                                (reward_a, done_a, truncated_a),
                                (reward_b, done_b, truncated_b),
                            )
                            self.assert_state_equal(
                                upstream.unwrapped.get_state_dict(),
                                headless.unwrapped.get_state_dict(),
                            )
                            np.testing.assert_array_equal(
                                extract_state_from_environment(
                                    peg_insertion_side_env=upstream.unwrapped
                                ),
                                extract_state_from_environment(
                                    peg_insertion_side_env=headless.unwrapped
                                ),
                            )
                    self.assertFalse(torch.equal(geometries[0], geometries[1]))
                    self.assertTrue(torch.equal(geometries[0], geometries[2]))
        finally:
            torch.set_num_threads(thread_count)


if __name__ == "__main__":
    unittest.main()
