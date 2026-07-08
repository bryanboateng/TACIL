import torch

from scripts.data.augmentation.tacil.models.dynamics_model import DynamicsModel
from scripts.data.augmentation.tacil.models.normalizers import Normalizers


class DynamicsModelEnsemble(torch.nn.Module):
    def __init__(
        self, *, normalizers: Normalizers, dynamics_models: list[DynamicsModel]
    ):
        super().__init__()
        self.normalizers = normalizers
        self._dynamics_models = torch.nn.ModuleList(modules=dynamics_models)

    @property
    def state_dimensionality(self) -> int:
        return self._first_dynamics_model.state_dimensionality

    @property
    def action_dimensionality(self) -> int:
        return self._first_dynamics_model.action_dimensionality

    @property
    def _first_dynamics_model(self) -> DynamicsModel:
        first_dynamics_model = self._dynamics_models[0]
        assert isinstance(first_dynamics_model, DynamicsModel)
        return first_dynamics_model

    def normalize_state(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        return self.normalizers.normalize_state(state_batch=state_batch)

    def denormalize_state(
        self,
        *,
        normalized_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        return self.normalizers.denormalize_state(
            normalized_state_batch=normalized_state_batch
        )

    def predict_next_state(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        return self.predict_next_state_per_member(
            state_batch=state_batch, action_batch=action_batch
        ).mean(dim=0)

    def predict_next_state_per_member(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (member_count, batch_size, state_dimensionality)
        return torch.stack(
            tensors=[
                dynamics_model.predict_next_state(
                    normalizers=self.normalizers,
                    state_batch=state_batch,
                    action_batch=action_batch,
                )
                for dynamics_model in self.dynamics_models
            ],
            dim=0,
        )

    def local_lipschitz_constants(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size,)
        return torch.stack(
            tensors=[
                dynamics_model.local_lipschitz_constants(
                    normalizers=self.normalizers,
                    state_batch=state_batch,
                    action_batch=action_batch,
                )
                for dynamics_model in self.dynamics_models
            ],
            dim=0,
        ).mean(dim=0)

    @property
    def dynamics_models(self) -> list[DynamicsModel]:
        # ModuleList is typed as nn.Module but only ever holds DynamicsNetworks.
        dynamics_models: list[DynamicsModel] = []
        for dynamics_model in self._dynamics_models:
            assert isinstance(dynamics_model, DynamicsModel)
            dynamics_models.append(dynamics_model)
        return dynamics_models
