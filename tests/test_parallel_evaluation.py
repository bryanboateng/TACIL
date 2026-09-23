import multiprocessing
import unittest
from functools import partial

import gymnasium as gym
import numpy as np

from scripts.evaluation.core.parallel import ParallelEnvironmentPool
from scripts.evaluation.core.rollout import roll_out_episodes


class ToyEnvironment(gym.Env):
    def __init__(self, *, failure=None):
        if failure == "create":
            raise RuntimeError("intentional creation failure")
        self.failure = failure
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)
        self.done = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.seed = seed
        self.steps = 0
        self.value = self.np_random.normal()
        self.done = False
        return self.observation(), {}

    def observation(self):
        return np.array([self.seed, self.steps, self.value], dtype=np.float32)

    def step(self, action):
        if self.failure == "step":
            raise RuntimeError("intentional step failure")
        if self.done:
            raise AssertionError("A completed episode was stepped again")
        self.steps += 1
        self.value += float(action[0])
        self.done = self.steps == self.seed % 3 + 1
        return (
            self.observation(),
            self.value,
            self.done and self.seed % 2 == 0,
            self.done and self.seed % 2 != 0,
            {},
        )


def seeded_policy_factory(*, episode_seeds):
    generators = [np.random.default_rng(seed) for seed in episode_seeds]

    def policy(*, observations_by_episode_index):
        return {
            i: generators[i].uniform(-1, 1, size=1).astype(np.float32)
            for i in observations_by_episode_index
        }

    return policy


class ParallelEvaluationTests(unittest.TestCase):
    def assert_no_new_workers(self, before):
        self.assertEqual({p.pid for p in multiprocessing.active_children()}, before)

    def test_parallel_matches_serial_with_early_end_and_partial_batch(self):
        seeds = list(range(10, 17))
        envs = [ToyEnvironment() for _ in seeds]
        try:
            expected = roll_out_episodes(
                environments=envs,
                seeds=seeds,
                policy=seeded_policy_factory(episode_seeds=seeds),
                render=False,
            )
        finally:
            for env in envs:
                env.close()
        before = {p.pid for p in multiprocessing.active_children()}
        with ParallelEnvironmentPool(factory=ToyEnvironment, worker_count=2) as pool:
            self.assertEqual(len({p.pid for p in pool._workers}), 2)
            actual = pool.roll_out(seeds=seeds, policy_factory=seeded_policy_factory)
        self.assert_no_new_workers(before)
        for left, right in zip(expected, actual, strict=True):
            self.assertEqual(
                (left.seed, left.reward, left.terminated, left.rewards),
                (right.seed, right.reward, right.terminated, right.rewards),
            )
            np.testing.assert_array_equal(
                left.final_observation, right.final_observation
            )
            np.testing.assert_array_equal(left.actions, right.actions)

    def test_worker_creation_failure_is_propagated_and_cleaned_up(self):
        before = {p.pid for p in multiprocessing.active_children()}
        with self.assertRaisesRegex(RuntimeError, "intentional creation failure"):
            ParallelEnvironmentPool(
                factory=partial(ToyEnvironment, failure="create"), worker_count=2
            )
        self.assert_no_new_workers(before)

    def test_worker_step_failure_is_propagated_and_cleaned_up(self):
        before = {p.pid for p in multiprocessing.active_children()}
        with self.assertRaisesRegex(RuntimeError, "intentional step failure"):
            with ParallelEnvironmentPool(
                factory=partial(ToyEnvironment, failure="step"), worker_count=2
            ) as pool:
                pool.roll_out(seeds=[10, 11], policy_factory=seeded_policy_factory)
        self.assert_no_new_workers(before)

    def test_parent_policy_failure_cleans_up_workers(self):
        before = {p.pid for p in multiprocessing.active_children()}

        def broken_factory(**kwargs):
            raise RuntimeError("intentional policy failure")

        with self.assertRaisesRegex(RuntimeError, "intentional policy failure"):
            with ParallelEnvironmentPool(
                factory=ToyEnvironment, worker_count=2
            ) as pool:
                pool.roll_out(seeds=[10, 11], policy_factory=broken_factory)
        self.assert_no_new_workers(before)


if __name__ == "__main__":
    unittest.main()
