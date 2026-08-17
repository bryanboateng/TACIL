import torch

from scripts.data.augmentation.tacil.dynamics.dynamics_model.dynamics_model import (
    DynamicsModel,
)
from scripts.shared.normalization import ZScoreNormalizer
from scripts.shared.pusht_state import average_states, subtract_states


class DynamicsModelEnsemble(torch.nn.Module):
    def __init__(
        self,
        *,
        state_normalizer: ZScoreNormalizer,
        dynamics_models: list[DynamicsModel],
    ):
        super().__init__()
        self.state_normalizer = state_normalizer
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
    def replay_error(
        self,
        *,
        start_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch_list: list[
            torch.Tensor
        ],  # each (batch_size, action_dimensionality), first one applied first
        end_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    ) -> torch.Tensor:  # (batch_size,)
        """RMS distance of the members' replay endpoints from the claimed end.

        Each member plays the whole action chain forward on its own, so an
        error made early passes through every later transition — exactly what
        happens when the chain is executed.

        Measured against the claimed end, not against the members' own mean, so
        it grows both when the members disagree and when they agree on the
        wrong endpoint.
        """
        per_member_end_state_batch = torch.stack(
            tensors=[
                _replay_chain(
                    dynamics_model=dynamics_model,
                    start_state_batch=start_state_batch,
                    chain_action_batches=action_batch_list,
                )
                for dynamics_model in self.dynamics_models
            ],
            dim=0,
        )  # (member_count, batch_size, state_dimensionality)

        per_member_deviation_batch = self.state_normalizer.normalize_difference(
            difference_batch=subtract_states(
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


def _replay_chain(
    *,
    dynamics_model: DynamicsModel,
    start_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    chain_action_batches: list[torch.Tensor],
) -> torch.Tensor:  # (batch_size, state_dimensionality)
    state_batch = start_state_batch
    for action_batch in chain_action_batches:
        state_batch = dynamics_model.predict_next_state(
            state_batch=state_batch, action_batch=action_batch
        )
    return state_batch
