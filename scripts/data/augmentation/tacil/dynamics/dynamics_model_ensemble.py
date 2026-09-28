from collections.abc import Sequence

import torch

from scripts.data.augmentation.tacil.dynamics.ensemble_network import (
    calculate_spectral_norms,
    create_ensemble_network,
)
from scripts.data.augmentation.tacil.dynamics.normalizers import Normalizers
from scripts.data.augmentation.tacil.dynamics.state_distance import (
    calculate_normalized_state_difference,
)
from scripts.environments.state_space import StateSpace
from scripts.environments.tacil_network_input import TacilNetworkInputBuilder


class DynamicsModelEnsemble(torch.nn.Module):
    def __init__(
        self,
        *,
        member_count: int,
        state_space: StateSpace,
        network_input_builder: TacilNetworkInputBuilder,
        hidden_layer_count: int,
        hidden_layer_dimensionality: int,
        lipschitz_limit: float | None,
        normalizers: Normalizers,
    ):
        super().__init__()

        self.member_count = member_count
        self.state_space = state_space
        self.network_input_builder = network_input_builder
        self.hidden_layer_count = hidden_layer_count
        self.hidden_layer_dimensionality = hidden_layer_dimensionality
        self.lipschitz_limit = lipschitz_limit
        self.normalizers = normalizers

        self._network = create_ensemble_network(
            member_count=member_count,
            input_dimensionality=normalizers.network_input.mean.shape[0],
            output_dimensionality=state_space.dimensionality,
            hidden_layer_count=hidden_layer_count,
            hidden_layer_dimensionality=hidden_layer_dimensionality,
            lipschitz_limit=lipschitz_limit,
        )

    def predict_next_state(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        return self.state_space.average(
            state_batches=self.predict_next_state_per_member(
                state_batch=state_batch, action_batch=action_batch
            )
        )

    def predict_next_state_per_member(
        self,
        *,
        state_batch: torch.Tensor,
        # (member_count, batch_size, state_dimensionality), or
        # (batch_size, state_dimensionality) for one batch every member shares
        action_batch: torch.Tensor,
        # (member_count, batch_size, action_dimensionality), or
        # (batch_size, action_dimensionality) for one batch every member shares
    ) -> torch.Tensor:  # (member_count, batch_size, state_dimensionality)
        normalized_state_displacement_batch = self._network(
            self._create_normalized_network_input(
                state_batch=state_batch, action_batch=action_batch
            )
        )  # (member_count, batch_size, state_dimensionality)

        state_displacement_batch = torch.where(
            condition=~self.normalizers.state_displacement.is_constant,
            input=self.normalizers.state_displacement.denormalize(
                value_batch=normalized_state_displacement_batch
            ),
            other=0.0,
        )  # (member_count, batch_size, state_dimensionality)

        return self.state_space.add(
            first_addend_batch=state_batch, second_addend_batch=state_displacement_batch
        )

    def calculate_prediction_uncertainty(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size,)
        per_member_state_batch = self.predict_next_state_per_member(
            state_batch=state_batch, action_batch=action_batch
        )  # (member_count, batch_size, state_dimensionality)
        return self._calculate_member_spread(
            per_member_state_batch=per_member_state_batch,
            reference_state_batch=self.state_space.average(
                state_batches=per_member_state_batch
            ),
        )

    def calculate_replay_error(
        self,
        *,
        start_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch_list: list[
            torch.Tensor
        ],  # each (batch_size, action_dimensionality), first one applied first
        end_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    ) -> torch.Tensor:  # (batch_size,)
        per_member_state_batch = start_state_batch
        for action_batch in action_batch_list:
            per_member_state_batch = self.predict_next_state_per_member(
                state_batch=per_member_state_batch, action_batch=action_batch
            )  # (member_count, batch_size, state_dimensionality)

        return self._calculate_member_spread(
            per_member_state_batch=per_member_state_batch,
            reference_state_batch=end_state_batch,
        )

    def calculate_local_lipschitz_constants(
        self,
        *,
        state_batch: torch.Tensor,  # (member_count, batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (member_count, batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (member_count, batch_size)
        jacobian_batch = self._network.calculate_jacobians(
            input_batch=self._create_normalized_network_input(
                state_batch=state_batch, action_batch=action_batch
            )
        )  # (member_count, batch_size, state_dimensionality, network_input_dimensionality)

        return calculate_spectral_norms(matrix_batch=jacobian_batch)

    # ---- Private --------------------------------------------

    def _create_normalized_network_input(
        self,
        *,
        state_batch: torch.Tensor,  # (*batch_shape, state_dimensionality)
        action_batch: torch.Tensor,  # (*batch_shape, action_dimensionality)
    ) -> torch.Tensor:  # (*batch_shape, network_input_dimensionality)
        batch_shape = torch.broadcast_shapes(
            state_batch.shape[:-1], action_batch.shape[:-1]
        )
        return self.normalizers.network_input.normalize(
            value_batch=self.network_input_builder(
                state_batch=state_batch.expand(*batch_shape, state_batch.shape[-1]),
                action_batch=action_batch.expand(*batch_shape, action_batch.shape[-1]),
            )
        )

    def _calculate_member_spread(
        self,
        *,
        per_member_state_batch: torch.Tensor,  # (member_count, batch_size, state_dimensionality)
        reference_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    ) -> torch.Tensor:  # (batch_size,)
        per_member_deviation_batch = calculate_normalized_state_difference(
            minuend_batch=per_member_state_batch,
            subtrahend_batch=reference_state_batch.unsqueeze(dim=0),
            state_space=self.state_space,
            state_normalizer=self.normalizers.state,
        )  # (member_count, batch_size, state_dimensionality)

        per_dimension_deviation_batch = (
            per_member_deviation_batch.square().mean(dim=0).sqrt()
        )  # (batch_size, state_dimensionality)

        return torch.linalg.norm(per_dimension_deviation_batch, dim=-1)  # (batch_size,)


def select_members(
    *, dynamics_model_ensemble: DynamicsModelEnsemble, member_indices: Sequence[int]
) -> DynamicsModelEnsemble:
    selected_ensemble = DynamicsModelEnsemble(
        member_count=len(member_indices),
        state_space=dynamics_model_ensemble.state_space,
        network_input_builder=dynamics_model_ensemble.network_input_builder,
        hidden_layer_count=dynamics_model_ensemble.hidden_layer_count,
        hidden_layer_dimensionality=dynamics_model_ensemble.hidden_layer_dimensionality,
        lipschitz_limit=dynamics_model_ensemble.lipschitz_limit,
        normalizers=dynamics_model_ensemble.normalizers,
    ).to(device=next(dynamics_model_ensemble.parameters()).device)

    with torch.no_grad():
        for selected_parameter, parameter in zip(
            selected_ensemble.parameters(),
            dynamics_model_ensemble.parameters(),
            strict=True,
        ):
            selected_parameter.copy_(parameter[list(member_indices)])

    return selected_ensemble
