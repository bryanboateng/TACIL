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

    @torch.no_grad()
    def disagreements_about_next_state(
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

        member_count, batch_size, state_dimensionality = (
            per_member_next_state_batch.shape
        )

        # normalize_state expects a 2D input of shape (state_count, state_dimensionality),
        # but we have a 3D tensor of per-member, per-batch-item states.
        # So we temporarily merge the member and batch dimensions into one,
        # normalize everything at once, then split them back apart.
        normalized_per_member_next_state_batch = self.normalize_state(
            state_batch=per_member_next_state_batch.flatten(
                start_dim=0, end_dim=1
            )  # (member_count * batch_size, state_dimensionality)
        ).reshape(
            member_count, batch_size, state_dimensionality
        )  # (member_count, batch_size, state_dimensionality)

        # For each example in the batch and each state dimension, compute how
        # spread out the ensemble's predictions are.
        per_dimension_disagreement_batch = normalized_per_member_next_state_batch.std(
            dim=0, unbiased=False
        )  # (batch_size, state_dimensionality)

        # Collapse the per-dimension disagreement into a single number per example
        return torch.linalg.norm(
            per_dimension_disagreement_batch, dim=-1
        )  # (batch_size,)

    @torch.no_grad()
    def sensitivity_scaled_step_distances(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        next_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size,)
        """Step size times local steepness: the room a step leaves for error
        the ensemble cannot see. The steepness is evaluated at the successor —
        in the backward rollout that is the trustworthy side of the transition,
        the one anchored by data."""
        local_lipschitz_constant_batch = self.local_lipschitz_constants(
            state_batch=next_state_batch, action_batch=action_batch
        )  # (batch_size,)

        normalized_step_distance_batch: torch.Tensor = torch.linalg.norm(
            self.normalize_state(state_batch=state_batch)
            - self.normalize_state(state_batch=next_state_batch),
            dim=-1,
        )  # (batch_size,)

        return (
            local_lipschitz_constant_batch * normalized_step_distance_batch
        )  # (batch_size,)

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

    @property
    def dynamics_models(self) -> list[DynamicsModel]:
        # ModuleList is typed as nn.Module but only ever holds DynamicsNetworks.
        dynamics_models: list[DynamicsModel] = []
        for dynamics_model in self._dynamics_models:
            assert isinstance(dynamics_model, DynamicsModel)
            dynamics_models.append(dynamics_model)
        return dynamics_models
