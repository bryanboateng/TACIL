import math
import unittest

import numpy as np
import torch
from torch.nn import functional

from scripts.data.augmentation.tacil.create_dynamics.dynamics_model_training import (
    TrainingContext,
    create_training_state,
    train_for_one_epoch,
)
from scripts.data.augmentation.tacil.dynamics.normalizers import Normalizers
from scripts.data.augmentation.tacil.dynamics.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)
from scripts.data.augmentation.tacil.dynamics.ensemble_network import (
    _calculate_lipschitz_limit_per_layer,
)
from scripts.environments.state_space import StateSpace
from scripts.shared.normalization import ZScoreNormalizer

STATE_DIMENSIONALITY = 4
ACTION_DIMENSIONALITY = 2
MEMBER_COUNT = 3
LIPSCHITZ_LIMIT = 2.0


def build_network_input(*, state_batch, action_batch):
    return torch.cat(tensors=[state_batch, action_batch], dim=-1)


def create_ensemble(*, lipschitz_limit=LIPSCHITZ_LIMIT):
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    state_batch = torch.randn(200, STATE_DIMENSIONALITY, generator=generator)
    action_batch = torch.randn(200, ACTION_DIMENSIONALITY, generator=generator)
    ensemble = DynamicsModelEnsemble(
        member_count=MEMBER_COUNT,
        state_space=StateSpace(dimensionality=STATE_DIMENSIONALITY, angle_indices=(3,)),
        network_input_builder=build_network_input,
        hidden_layer_count=2,
        hidden_layer_dimensionality=16,
        lipschitz_limit=lipschitz_limit,
        normalizers=Normalizers(
            state=ZScoreNormalizer(value_batch=state_batch),
            network_input=ZScoreNormalizer(
                value_batch=build_network_input(
                    state_batch=state_batch, action_batch=action_batch
                )
            ),
            state_displacement=ZScoreNormalizer(value_batch=0.1 * state_batch),
        ),
    )
    with torch.no_grad():
        for parameter in ensemble.parameters():
            parameter.mul_(3.0)
    return ensemble.eval()


def collect_member_layers(*, ensemble, member_index):
    parameters = dict(ensemble.named_parameters())
    layers = []
    for layer_index in range(3):
        prefix = f"_network._layers.{layer_index}"
        weight = parameters[f"{prefix}.parametrizations.weight.original"][member_index]
        weight = weight / torch.clamp(
            torch.linalg.matrix_norm(weight, ord=2)
            / _calculate_lipschitz_limit_per_layer(
                lipschitz_limit=ensemble.lipschitz_limit, linear_layer_count=3
            ),
            min=1.0,
        )
        layers.append((weight, parameters[f"{prefix}.bias"][member_index]))
    return layers


def apply_member_layers(*, layers, network_input):
    output = network_input
    for layer_index, (weight, bias) in enumerate(layers):
        output = functional.linear(output, weight, bias)
        if layer_index != len(layers) - 1:
            output = functional.silu(output)
    return output


class DynamicsModelEnsembleTests(unittest.TestCase):
    def setUp(self):
        self.ensemble = create_ensemble()
        generator = torch.Generator().manual_seed(1)
        self.state_batch = torch.randn(8, STATE_DIMENSIONALITY, generator=generator)
        self.action_batch = torch.randn(8, ACTION_DIMENSIONALITY, generator=generator)

    def test_per_member_prediction_matches_sequential_member_network(self):
        with torch.no_grad():
            per_member_prediction = self.ensemble.predict_next_state_per_member(
                state_batch=self.state_batch, action_batch=self.action_batch
            )
        self.assertEqual(
            per_member_prediction.shape, (MEMBER_COUNT, 8, STATE_DIMENSIONALITY)
        )

        network_input = self.ensemble.normalizers.network_input.normalize(
            value_batch=build_network_input(
                state_batch=self.state_batch, action_batch=self.action_batch
            )
        )
        for member_index in range(MEMBER_COUNT):
            with torch.no_grad():
                expected_prediction = self.ensemble.state_space.add(
                    first_addend_batch=self.state_batch,
                    second_addend_batch=self.ensemble.normalizers.state_displacement.denormalize(
                        value_batch=apply_member_layers(
                            layers=collect_member_layers(
                                ensemble=self.ensemble, member_index=member_index
                            ),
                            network_input=network_input,
                        )
                    ),
                )
            torch.testing.assert_close(
                per_member_prediction[member_index], expected_prediction
            )

    def test_local_lipschitz_constants_match_autograd_jacobian(self):
        member_index, sample_index = 1, 3
        constants = self.ensemble.calculate_local_lipschitz_constants(
            state_batch=self.state_batch.expand(MEMBER_COUNT, -1, -1),
            action_batch=self.action_batch.expand(MEMBER_COUNT, -1, -1),
        )
        self.assertEqual(constants.shape, (MEMBER_COUNT, 8))

        layers = collect_member_layers(
            ensemble=self.ensemble, member_index=member_index
        )
        network_input = self.ensemble.normalizers.network_input.normalize(
            value_batch=build_network_input(
                state_batch=self.state_batch, action_batch=self.action_batch
            )
        )[sample_index]
        jacobian = torch.autograd.functional.jacobian(
            lambda single_input: apply_member_layers(
                layers=layers, network_input=single_input
            ),
            network_input,
        )
        torch.testing.assert_close(
            constants[member_index, sample_index],
            torch.linalg.matrix_norm(jacobian, ord=2),
        )


class DivergedMemberIsolationTests(unittest.TestCase):
    def _create_training_state_and_context(self):
        generator = torch.Generator().manual_seed(3)
        transition_count = 300
        states = torch.randn(
            transition_count, STATE_DIMENSIONALITY, generator=generator
        )
        actions = torch.randn(
            transition_count, ACTION_DIMENSIONALITY, generator=generator
        )
        next_states = states + 0.1 * actions.sum(dim=-1, keepdim=True)
        state_windows = torch.stack(
            [
                torch.stack([states[index], next_states[index], next_states[index + 1]])
                for index in range(transition_count - 1)
            ]
        )
        action_windows = torch.stack(
            [
                torch.stack([actions[index], actions[index + 1]])
                for index in range(transition_count - 1)
            ]
        )
        training_state = create_training_state(
            model=create_ensemble(), seed=3, training_window_count=transition_count - 1
        )
        context = TrainingContext(
            training_state_windows=state_windows,
            training_action_windows=action_windows,
            validation_state_windows=state_windows[:64],
            validation_action_windows=action_windows[:64],
        )
        return training_state, context

    def test_nan_member_does_not_affect_other_members(self):
        clean_state, context = self._create_training_state_and_context()
        poisoned_state, _ = self._create_training_state_and_context()
        with torch.no_grad():
            for parameter in poisoned_state.model.parameters():
                parameter[1] = math.nan

        for epoch_index in range(2):
            train_for_one_epoch(
                training_state=clean_state, epoch_index=epoch_index, context=context
            )
            poisoned_metrics = train_for_one_epoch(
                training_state=poisoned_state, epoch_index=epoch_index, context=context
            )

        for clean_model, poisoned_model in (
            (clean_state.model, poisoned_state.model),
            (clean_state.best_model, poisoned_state.best_model),
        ):
            for clean_parameter, poisoned_parameter in zip(
                clean_model.parameters(), poisoned_model.parameters(), strict=True
            ):
                self.assertTrue(
                    torch.equal(clean_parameter[[0, 2]], poisoned_parameter[[0, 2]])
                )
        self.assertTrue(np.isnan(poisoned_metrics.validation_losses[1]))
        self.assertTrue(np.isfinite(poisoned_metrics.validation_losses[[0, 2]]).all())
        self.assertTrue(np.isinf(poisoned_state.best_validation_losses[1]))
        self.assertTrue(
            np.isfinite(poisoned_state.best_validation_losses[[0, 2]]).all()
        )


if __name__ == "__main__":
    unittest.main()
