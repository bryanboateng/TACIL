import torch

from scripts.data.augmentation.tacil.dynamics.dynamics_model.dynamics_model import (
    DynamicsModel,
)
from scripts.data.augmentation.tacil.dynamics.state_distance import (
    calculate_normalized_state_difference,
)
from scripts.shared.normalization import ZScoreNormalizer
from scripts.shared.pusht_state import average_states


class DynamicsModelEnsemble(torch.nn.Module):
    def __init__(
        self,
        *,
        state_normalizer: ZScoreNormalizer,
        dynamics_models: list[DynamicsModel],
    ):
        super().__init__()
        self.state_normalizer = state_normalizer
        self._dynamics_models_module_list = torch.nn.ModuleList(modules=dynamics_models)

    @property
    def dynamics_models(self) -> tuple[DynamicsModel, ...]:
        members: list[DynamicsModel] = []
        for module in self._dynamics_models_module_list:
            if not isinstance(module, DynamicsModel):
                raise TypeError(
                    f"Expected a DynamicsModel member, got {type(module).__name__}"
                )
            members.append(module)
        return tuple(members)

    def predict_next_state(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        return average_states(
            state_batch=self._predict_next_state_per_member(
                state_batch=state_batch, action_batch=action_batch
            )
        )

    def calculate_prediction_uncertainty(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size,)
        per_member_state_batch = self._predict_next_state_per_member(
            state_batch=state_batch, action_batch=action_batch
        )  # (member_count, batch_size, state_dimensionality)
        return self._calculate_member_spread(
            per_member_state_batch=per_member_state_batch,
            reference_state_batch=average_states(state_batch=per_member_state_batch),
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
        per_member_end_state_batch = torch.stack(
            tensors=[
                _replay_actions(
                    dynamics_model=dynamics_model,
                    start_state_batch=start_state_batch,
                    action_batches=action_batch_list,
                )
                for dynamics_model in self.dynamics_models
            ],
            dim=0,
        )  # (member_count, batch_size, state_dimensionality)

        return self._calculate_member_spread(
            per_member_state_batch=per_member_end_state_batch,
            reference_state_batch=end_state_batch,
        )

    # ---- Private --------------------------------------------

    def _predict_next_state_per_member(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (member_count, batch_size, state_dimensionality)
        return torch.stack(
            tensors=[
                dynamics_model.predict_next_state(
                    state_batch=state_batch, action_batch=action_batch
                )
                for dynamics_model in self.dynamics_models
            ],
            dim=0,
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
            state_normalizer=self.state_normalizer,
        )  # (member_count, batch_size, state_dimensionality)

        per_dimension_deviation_batch = (
            per_member_deviation_batch.square().mean(dim=0).sqrt()
        )  # (batch_size, state_dimensionality)

        return torch.linalg.norm(per_dimension_deviation_batch, dim=-1)  # (batch_size,)


# ---- Private --------------------------------------------


def _replay_actions(
    *,
    dynamics_model: DynamicsModel,
    start_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    action_batches: list[torch.Tensor],
) -> torch.Tensor:  # (batch_size, state_dimensionality)
    state_batch = start_state_batch
    for action_batch in action_batches:
        state_batch = dynamics_model.predict_next_state(
            state_batch=state_batch, action_batch=action_batch
        )
    return state_batch
