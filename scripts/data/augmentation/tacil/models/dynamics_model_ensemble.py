import torch

from scripts.data.augmentation.tacil.models.dynamics_model.dynamics_model import (
    DynamicsModel,
)
from scripts.data.augmentation.tacil.models.normalizers import Normalizers
from scripts.shared.pusht_state import average_states, subtract_states


class DynamicsModelEnsemble(torch.nn.Module):
    def __init__(
        self, *, normalizers: Normalizers, dynamics_models: list[DynamicsModel]
    ):
        super().__init__()
        self.normalizers = normalizers
        self._dynamics_models = torch.nn.ModuleList(modules=dynamics_models)

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

    @torch.no_grad()
    def uncertainty_about_next_state(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size,)
        per_member_next_state_batch = self._predict_next_state_per_member(
            state_batch=state_batch, action_batch=action_batch
        )  # (member_count, batch_size, state_dimensionality)

        average_next_state_batch = average_states(
            state_batch=per_member_next_state_batch
        )  # (batch_size, state_dimensionality)

        per_member_deviation_batch = (
            self.normalizers.state_normalizer.normalize_difference(
                difference_batch=subtract_states(
                    minuend_batch=per_member_next_state_batch,
                    subtrahend_batch=average_next_state_batch.unsqueeze(dim=0),
                )
            )
        )  # (member_count, batch_size, state_dimensionality)

        per_dimension_uncertainty_batch = (
            per_member_deviation_batch.square().mean(dim=0).sqrt()
        )  # (batch_size, state_dimensionality)

        return torch.linalg.norm(
            per_dimension_uncertainty_batch, dim=-1
        )  # (batch_size,)

    @property
    def dynamics_models(self) -> list[DynamicsModel]:
        dynamics_models: list[DynamicsModel] = []
        for dynamics_model in self._dynamics_models:
            assert isinstance(dynamics_model, DynamicsModel)
            dynamics_models.append(dynamics_model)
        return dynamics_models

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
