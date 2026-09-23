import unittest
from unittest.mock import Mock

import numpy as np
import torch

from scripts.evaluation.core.rollout import EpisodeRecord
from scripts.evaluation.core.scoring import pick_representative_episodes
from scripts.policies.bc.config import Config, OptimizationConfig
from scripts.policies.bc.train import train_behavior_cloning_policy
from scripts.shared.normalization import ZScoreNormalizer
from scripts.shared.state_action_pairs import StateActionPairs


class NormalizationTests(unittest.TestCase):
    def test_float32_roundoff_is_not_amplified(self):
        values = torch.tensor([[0.0, 1.0], [1e-8, 2.0], [-1e-8, 3.0]])
        normalizer = ZScoreNormalizer(value_batch=values)
        normalized = normalizer.normalize(value_batch=values)
        self.assertLess(normalized[:, 0].abs().max().item(), 1e-6)
        self.assertAlmostEqual(normalized[:, 1].std(correction=0).item(), 1.0, places=6)
        torch.testing.assert_close(
            normalizer.denormalize(value_batch=normalized), values
        )

    def test_single_sample_is_finite(self):
        values = torch.tensor([[1.0, 2.0]])
        normalizer = ZScoreNormalizer(value_batch=values)
        self.assertTrue(torch.isfinite(normalizer.normalize(value_batch=values)).all())


class TrainingTests(unittest.TestCase):
    def test_partial_batch_and_epoch_logging(self):
        rng = np.random.default_rng(0)
        states = rng.normal(size=(17, 3)).astype(np.float32)
        actions = states[:, :1] * 0.2
        run = Mock(summary={})
        model = train_behavior_cloning_policy(
            config=Config(
                optimization=OptimizationConfig(batch_size=8, epoch_count=2),
                hidden_layer_dimensionality=8,
                validation=None,
                excluded_observation_indices=(0,),
            ),
            environment=None,
            state_action_pairs=StateActionPairs(states=states, actions=actions),
            seed=0,
            device=torch.device("cpu"),
            run=run,
        )
        self.assertEqual(run.log.call_count, 2)
        self.assertEqual(run.log.call_args.kwargs["data"]["3_policy/0_step"], 6)
        self.assertTrue(torch.isfinite(model(torch.from_numpy(states))).all())
        altered_states = torch.from_numpy(states.copy())
        altered_states[:, 0] += 100
        torch.testing.assert_close(
            model(torch.from_numpy(states)), model(altered_states)
        )
        # CCIL replication exports this same model through TorchScript.
        scripted_model = torch.jit.script(model)
        torch.testing.assert_close(
            scripted_model(altered_states), model(torch.from_numpy(states))
        )


class VideoSelectionTests(unittest.TestCase):
    def test_no_episode_is_selected_twice(self):
        records = [
            EpisodeRecord(
                seed=i, reward=float(i), terminated=False, final_observation=np.zeros(1)
            )
            for i in range(10)
        ]
        selected = pick_representative_episodes(episode_records=records)
        seeds = [r.seed for r in selected.values()]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertIn(0, seeds)
        self.assertIn(9, seeds)
        limited = pick_representative_episodes(
            episode_records=records, maximum_episode_count=3
        )
        self.assertEqual(list(limited), [0, 50, 100])
        self.assertEqual(len(limited), 3)


if __name__ == "__main__":
    unittest.main()
