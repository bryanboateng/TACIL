import torch

from scripts.data.augmentation.tacil.models.dynamics_model import DynamicsModel
from scripts.data.augmentation.tacil.models.normalizers import Normalizers
from scripts.shared.pusht_state import average_states, subtract_states


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

    def predict_next_state(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        return average_states(
            state_batch=self.predict_next_state_per_member(
                state_batch=state_batch, action_batch=action_batch
            )
        )

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

    @torch.no_grad()
    def uncertainty_about_next_state(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size,)
        """How much the ensemble members disagree about the next state."""

        # Ask every member to predict the next state for the same batch of
        # (state, action) pairs, independently.
        per_member_next_state_batch = self.predict_next_state_per_member(
            state_batch=state_batch, action_batch=action_batch
        )  # (member_count, batch_size, state_dimensionality)

        # The spread is measured around the ensemble's own mean prediction. That
        # mean has to treat the orientation as a direction, otherwise two
        # members agreeing across the 0 / 2π seam would average to the opposite
        # orientation and look like maximal disagreement.
        average_next_state_batch = average_states(
            state_batch=per_member_next_state_batch
        )  # (batch_size, state_dimensionality)

        deviation_batch = self.normalizers.scale_to_state_spread_units(
            state_difference_batch=subtract_states(
                minuend_batch=per_member_next_state_batch,
                subtrahend_batch=average_next_state_batch.unsqueeze(dim=0),
            )
        )  # (member_count, batch_size, state_dimensionality)

        # For each example in the batch and each state dimension, compute how
        # spread out the ensemble's predictions are.
        per_dimension_disagreement_batch = (
            deviation_batch.square().mean(dim=0).sqrt()
        )  # (batch_size, state_dimensionality)

        # Collapse the per-dimension disagreement into a single number per example
        return torch.linalg.norm(
            per_dimension_disagreement_batch, dim=-1
        )  # (batch_size,)

    def local_transition_lipschitz_constants(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size,)
        """Worst member's gain of the full transition map (identity included).

        This stands in for the local sensitivity of the TRUE dynamics, which
        decides how strongly errors compound when a synthetic prefix is
        executed forward. Underestimating it silently weakens the error
        account, while overestimating merely shortens prefixes — so take the
        maximum over members rather than the mean.
        """
        return (
            torch.stack(
                tensors=[
                    dynamics_model.local_transition_lipschitz_constants(
                        normalizers=self.normalizers,
                        state_batch=state_batch,
                        action_batch=action_batch,
                    )
                    for dynamics_model in self.dynamics_models
                ],
                dim=0,
            )
            .max(dim=0)
            .values
        )

    @torch.no_grad()
    def replay_landing_errors(
        self,
        *,
        start_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_window_batch: torch.Tensor,  # (batch_size, step_count, action_dimensionality)
        target_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    ) -> torch.Tensor:  # (batch_size,)
        """How far the members land from the target when they drive the given
        actions forward from the start state.

        This is the promise a synthetic prefix makes: a policy starts at the
        start state, executes the actions, and has to arrive at the target.
        Each member keeps its own state along the way, so a disagreement
        compounds through the real non-linear map instead of through a product
        of upper bounds."""
        step_count = action_window_batch.shape[1]

        landing_error_columns: list[torch.Tensor] = []
        for dynamics_model in self.dynamics_models:
            state_batch = start_state_batch
            for step_index in range(step_count):
                state_batch = dynamics_model.predict_next_state(
                    normalizers=self.normalizers,
                    state_batch=state_batch,
                    action_batch=action_window_batch[:, step_index, :],
                )
            landing_error_columns.append(
                torch.linalg.norm(
                    self.normalizers.scale_to_state_spread_units(
                        state_difference_batch=subtract_states(
                            minuend_batch=state_batch,
                            subtrahend_batch=target_state_batch,
                        )
                    ),
                    dim=-1,
                )
            )

        return torch.stack(tensors=landing_error_columns, dim=0).mean(dim=0)

    @property
    def dynamics_models(self) -> list[DynamicsModel]:
        dynamics_models: list[DynamicsModel] = []
        for dynamics_model in self._dynamics_models:
            assert isinstance(dynamics_model, DynamicsModel)
            dynamics_models.append(dynamics_model)
        return dynamics_models
