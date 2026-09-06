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
        self.dynamics_models = dynamics_models

        # Torch reaches parameters only through modules it was handed, so the
        # same members are registered a second time under a name it walks.
        self._registered_dynamics_models = torch.nn.ModuleList(modules=dynamics_models)

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
    def measure_replay_error(
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

        return self.measure_endpoint_deviation(
            per_member_state_batch=per_member_end_state_batch,
            target_state_batch=end_state_batch,
        )

    @torch.no_grad()
    def measure_endpoint_deviation(
        self,
        *,
        per_member_state_batch: torch.Tensor,  # (member_count, batch_size, state_dimensionality)  # noqa: E501
        target_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    ) -> torch.Tensor:  # (batch_size,)
        per_member_deviation_batch = self.state_normalizer.normalize_difference(
            difference_batch=subtract_states(
                minuend_batch=per_member_state_batch,
                subtrahend_batch=target_state_batch.unsqueeze(dim=0),
            )
        )  # (member_count, batch_size, state_dimensionality)

        per_dimension_deviation_batch = (
            per_member_deviation_batch.square().mean(dim=0).sqrt()
        )  # (batch_size, state_dimensionality)

        return torch.linalg.norm(per_dimension_deviation_batch, dim=-1)  # (batch_size,)

    @torch.no_grad()
    def measure_prediction_disagreement(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size,)
        """How far the members' predictions of one step lie apart.

        Measured the same way as the replay error, but around the members' own
        mean instead of a target: where the members agree the model has been
        told what happens here, and where they scatter it is extrapolating.
        """
        per_member_state_batch = self._predict_next_state_per_member(
            state_batch=state_batch, action_batch=action_batch
        )  # (member_count, batch_size, state_dimensionality)
        return self.measure_endpoint_deviation(
            per_member_state_batch=per_member_state_batch,
            target_state_batch=average_states(state_batch=per_member_state_batch),
        )

    @torch.no_grad()
    def measure_local_lipschitz_constants(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size,)
        """The members' average local Lipschitz constant: how much the model
        stretches space right here."""
        return torch.stack(
            tensors=[
                dynamics_model.measure_local_lipschitz_constants(
                    state_batch=state_batch, action_batch=action_batch
                )
                for dynamics_model in self.dynamics_models
            ],
            dim=0,
        ).mean(dim=0)

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
