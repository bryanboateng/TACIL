import torch

from scripts.data.augmentation.tacil.dynamics.dynamics_model.dynamics_model import (
    DynamicsModel,
)
from scripts.environments.state_space import StateSpace
from scripts.shared.normalization import ZScoreNormalizer


class DynamicsModelEnsemble(torch.nn.Module):
    def __init__(
        self,
        *,
        state_space: StateSpace,
        state_normalizer: ZScoreNormalizer,
        dynamics_models: list[DynamicsModel],
    ):
        super().__init__()
        self.state_space = state_space
        self.state_normalizer = state_normalizer
        self._dynamics_models = torch.nn.ModuleList(modules=dynamics_models)

    def predict_next_state(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        return self.state_space.average(
            state_batch=self._predict_next_state_per_member(
                state_batch=state_batch, action_batch=action_batch
            )
        )

    @torch.no_grad()
    def replay_error(
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

        per_member_deviation_batch = self.state_normalizer.normalize_difference(
            difference_batch=self.state_space.subtract(
                minuend_batch=per_member_end_state_batch,
                subtrahend_batch=end_state_batch.unsqueeze(dim=0),
            )
        )  # (member_count, batch_size, state_dimensionality)

        per_dimension_deviation_batch = (
            per_member_deviation_batch.square().mean(dim=0).sqrt()
        )  # (batch_size, state_dimensionality)

        return torch.linalg.norm(per_dimension_deviation_batch, dim=-1)  # (batch_size,)

    @property
    def dynamics_models(self) -> list[DynamicsModel]:
        dynamics_models: list[DynamicsModel] = []
        for dynamics_model in self._dynamics_models:
            assert isinstance(dynamics_model, DynamicsModel)
            dynamics_models.append(dynamics_model)
        return dynamics_models

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
