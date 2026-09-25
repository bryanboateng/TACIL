import os
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np

from scripts.evaluation.config import VALIDATION_START_SEED
from scripts.evaluation.core.scoring import (
    ScoringSession,
    roll_out_episodes_with_separate_seeds_per_condition,
    roll_out_episodes_with_the_same_seeds_in_every_condition,
)


@dataclass(frozen=True)
class ToyCondition:
    label: str


class ToyEnvironment(gym.Env):
    def __init__(self, condition, directory):
        self.condition = condition
        self.directory = Path(directory)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(3,))
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,))
        self._mark("created")

    def _mark(self, event):
        with (self.directory / f"{os.getpid()}_{self.condition.label}").open("a") as f:
            f.write(event + "\n")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.limit = 1 + seed % 3
        self.state = np.array(
            [float(self.condition.label), self.np_random.random(), 0], dtype=np.float32
        )
        return self.state.copy(), {}

    def step(self, action):
        if action[0] == 999:
            raise ValueError("test simulator failure")
        self.steps += 1
        self.state[1] += action[0]
        self.state[2] = self.steps
        return (
            self.state.copy(),
            float(self.state[1]),
            self.steps == self.limit,
            False,
            {},
        )

    def close(self):
        self._mark("closed")


@dataclass
class ToyTask:
    directory: str
    conditions = (ToyCondition("0"), ToyCondition("1"))

    def create_gym_environment(self, *, condition, render_settings):
        return ToyEnvironment(condition, self.directory)


def policy_factory(action):
    # Deliberately local/unpicklable: inference must remain in the parent.
    def factory(*, episode_seeds):
        steps = [0] * len(episode_seeds)

        def policy(*, observations_by_episode_index):
            actions = {}
            for i in observations_by_episode_index:
                actions[i] = np.array([action + steps[i]], dtype=np.float32)
                steps[i] += 1
            return actions

        return policy

    return factory


class ScoringSessionTests(unittest.TestCase):
    def assert_records_equal(self, left, right):
        self.assertEqual(len(left), len(right))
        for a, b in zip(left, right, strict=True):
            self.assertEqual(
                (a.seed, a.reward, a.terminated), (b.seed, b.reward, b.terminated)
            )
            np.testing.assert_array_equal(a.final_observation, b.final_observation)
            np.testing.assert_array_equal(a.actions, b.actions)
            self.assertEqual(a.rewards, b.rewards)

    def test_reuses_workers_without_changing_episodes(self):
        with tempfile.TemporaryDirectory() as directory:
            task = ToyTask(directory)
            with ScoringSession(
                environment=task, parallel_environment_count=2
            ) as session:
                first = session.roll_out_episodes_with_separate_seeds_per_condition(
                    policy_factory=policy_factory(0.25), episode_count_per_condition=3
                )
                files = list(Path(directory).iterdir())
                # Two processes, each owning both conditions, not four processes.
                self.assertEqual(len({p.name.split("_")[0] for p in files}), 2)
                self.assertEqual(len(files), 4)
                second = session.roll_out_episodes_with_separate_seeds_per_condition(
                    policy_factory=policy_factory(0.25), episode_count_per_condition=3
                )
                self.assert_records_equal(first, second)
                updated = session.roll_out_episodes_with_separate_seeds_per_condition(
                    policy_factory=policy_factory(0.5), episode_count_per_condition=3
                )
                self.assertNotEqual(first[0].reward, updated[0].reward)
            # Each condition opens its own environments, so every one that was
            # created is closed again by the time the session ends.
            for path in files:
                events = path.read_text().split()
                self.assertEqual(events.count("created"), events.count("closed"))
                self.assertGreater(events.count("created"), 0)
            with_one_worker = roll_out_episodes_with_separate_seeds_per_condition(
                environment=task,
                policy_factory=policy_factory(0.25),
                episode_count_per_condition=3,
                parallel_environment_count=1,
            )
            self.assert_records_equal(first, with_one_worker)
            self.assertEqual(
                [r.seed for r in first],
                list(range(VALIDATION_START_SEED, VALIDATION_START_SEED + 6)),
            )

    def test_cleanup_on_worker_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "test simulator failure"):
                with ScoringSession(
                    environment=ToyTask(directory), parallel_environment_count=2
                ) as session:
                    session.roll_out_episodes_with_separate_seeds_per_condition(
                        policy_factory=policy_factory(999),
                        episode_count_per_condition=2,
                    )
            files = list(Path(directory).iterdir())
            self.assertEqual(len(files), 2)
            for path in files:
                self.assertEqual(path.read_text(), "created\nclosed\n")

    def test_lazy_startup_and_closed_session(self):
        with tempfile.TemporaryDirectory() as directory:
            session = ScoringSession(
                environment=ToyTask(directory), parallel_environment_count=2
            )
            with session:
                self.assertEqual(list(Path(directory).iterdir()), [])
            session.close()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                session.roll_out_episodes_with_separate_seeds_per_condition(
                    policy_factory=policy_factory(0), episode_count_per_condition=2
                )

    def test_closing_partial_evaluation_releases_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            records = roll_out_episodes_with_the_same_seeds_in_every_condition(
                environment=ToyTask(directory),
                policy_factory=policy_factory(0),
                start_episode_seed=50,
                episode_count_per_condition=2,
                parallel_environment_count=2,
            )
            try:
                condition, episodes = next(records)
                self.assertEqual(condition.label, "0")
                self.assertEqual([record.seed for record in episodes], [50, 51])
            finally:
                records.close()
            files = list(Path(directory).iterdir())
            self.assertEqual(len(files), 2)
            for path in files:
                self.assertEqual(path.read_text(), "created\nclosed\n")


if __name__ == "__main__":
    unittest.main()
