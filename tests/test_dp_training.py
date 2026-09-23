import copy
import unittest
from collections import Counter
from unittest.mock import Mock, patch

import numpy as np
import torch

from scripts.evaluation.core.rollout import EpisodeRecord
from scripts.policies.dp.config import (
    Config,
    HorizonLengthConfig,
    OptimizationConfig,
    ValidationConfig,
)
from scripts.policies.dp.dataset import Dataset
from scripts.policies.dp.policy import create_diffusion_policy_factory
from scripts.policies.dp.train import train_diffusion_policy
from scripts.shared.episode_store import EpisodeStore


def episodes():
    rng = np.random.default_rng(0)
    states = rng.normal(size=(40, 3)).astype(np.float32)
    states[:, 0] *= 1e-8
    return EpisodeStore(
        states=states,
        actions=rng.uniform(-1, 1, size=(40, 2)).astype(np.float32),
        episode_end_indices=np.array([20, 40]),
    )


class DiffusionTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_observation_roundoff_and_shuffled_batches(self):
        dataset = Dataset(
            expert_episodes=episodes(),
            augmented_episodes=None,
            horizon_lengths=HorizonLengthConfig(),
        )
        self.assertEqual(dataset.observation_normalizer.scale[0].item(), 1.0)
        dataset.to(device=torch.device("cpu"))
        batches = list(dataset.batches(batch_size=7))
        self.assertEqual([len(obs) for obs, _ in batches], [7, 7, 7, 5])
        expected = Counter(
            tuple(dataset[i][1].flatten().tolist()) for i in range(len(dataset))
        )
        actual = Counter(
            tuple(action.flatten().tolist())
            for _, actions in batches
            for action in actions
        )
        self.assertEqual(actual, expected)
        self.assertLess(
            torch.cat([obs for obs, _ in batches])[:, :, 0].abs().max().item(), 1e-6
        )

    def test_epoch_metrics_validation_success_and_best_ema_checkpoint(self):
        snapshots = []

        def record_factory(**kwargs):
            snapshots.append(
                copy.deepcopy(kwargs["noise_prediction_network"].state_dict())
            )
            return create_diffusion_policy_factory(**kwargs)

        records = [
            [
                EpisodeRecord(
                    seed=1,
                    reward=reward,
                    terminated=success,
                    final_observation=np.zeros(3),
                )
            ]
            for reward, success in [(1.0, True), (0.1, False)]
        ]
        run = Mock(summary={})
        with (
            patch(
                "scripts.policies.dp.train.run_scored_episodes_over_conditions",
                side_effect=records,
            ) as score,
            patch(
                "scripts.policies.dp.train.create_diffusion_policy_factory",
                side_effect=record_factory,
            ),
        ):
            model = train_diffusion_policy(
                config=Config(
                    horizon_lengths=HorizonLengthConfig(),
                    optimization=OptimizationConfig(
                        batch_size=7, epoch_count=2, learning_rate_warmup_step_count=0
                    ),
                    validation=ValidationConfig(
                        interval_epoch_count=1,
                        episode_count_per_condition=1,
                        parallel_environment_count=2,
                    ),
                    hidden_channel_counts=(8, 16, 32),
                ),
                environment=Mock(),
                expert_episodes=episodes(),
                augmented_episodes=None,
                seed=0,
                device=torch.device("cpu"),
                run=run,
            )
        logs = [call.kwargs["data"] for call in run.log.call_args_list]
        self.assertEqual(len(logs), 4)
        self.assertEqual(logs[-1]["3_policy/0_step"], 8)
        self.assertEqual(logs[1]["3_policy/2_validation/3_success_rate"], 1.0)
        self.assertEqual(logs[-1]["3_policy/2_validation/3_success_rate"], 0.0)
        self.assertEqual(score.call_args.kwargs["parallel_environment_count"], 2)
        self.assertEqual(run.summary["3_policy/3_best_checkpoint/1_epoch"], 0)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, snapshots[0][name], rtol=0, atol=0)
        for sampler in ["ddim", "ddpm"]:
            policy = create_diffusion_policy_factory(
                noise_prediction_network=model,
                horizon_lengths=HorizonLengthConfig(),
                noise_scheduler_name=sampler,
                device=torch.device("cpu"),
            )(episode_seeds=[10])
            actions = policy(observations_by_episode_index={0: episodes().states[0]})
            self.assertTrue(np.isfinite(actions[0]).all())


if __name__ == "__main__":
    unittest.main()
