from typing import Self

import torch

from scripts.shared.normalization import ZScoreNormalizer

class Normalizers(torch.nn.Module):
    state_normalizer: ZScoreNormalizer
    action_normalizer: ZScoreNormalizer
    state_displacement_normalizer: ZScoreNormalizer

    def __init__(
        self,
        *,
        state_normalizer: ZScoreNormalizer,
        action_normalizer: ZScoreNormalizer,
        state_displacement_normalizer: ZScoreNormalizer,
    ):
        super().__init__()
        self.state_normalizer = state_normalizer
        self.action_normalizer = action_normalizer
        self.state_displacement_normalizer = state_displacement_normalizer

    @classmethod
    def from_transitions(
        cls,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
        next_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    ) -> Self:
        state_displacement_batch = next_state_batch - state_batch
        return cls(
            state_normalizer=ZScoreNormalizer.from_values(value_batch=state_batch),
            action_normalizer=ZScoreNormalizer.from_values(value_batch=action_batch),
            # Scale only, no shift. The shift would be the average motion in
            # the demonstrations, which is a property of the task (the expert
            # pushes the block toward one goal), not of the dynamics. As a
            # shift it becomes the model's default answer whenever the network
            # says nothing — so a block that is standing still, which it is in
            # most steps, would still drift in the expert's direction.
            state_displacement_normalizer=ZScoreNormalizer(
                mean=torch.zeros_like(input=state_displacement_batch[0]),
                std=state_displacement_batch.std(dim=0).clamp(min=1e-8),
            ),
        )

    @classmethod
    def placeholder(
        cls, *, state_dimensionality: int, action_dimensionality: int
    ) -> Self:
        """Identity normalizers whose buffers are overwritten by a state dict."""
        return cls(
            state_normalizer=ZScoreNormalizer(
                mean=torch.zeros(state_dimensionality),
                std=torch.ones(state_dimensionality),
            ),
            action_normalizer=ZScoreNormalizer(
                mean=torch.zeros(action_dimensionality),
                std=torch.ones(action_dimensionality),
            ),
            state_displacement_normalizer=ZScoreNormalizer(
                mean=torch.zeros(state_dimensionality),
                std=torch.ones(state_dimensionality),
            ),
        )

    def normalize_state(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        return self.state_normalizer.normalize(value_batch=state_batch)

    def denormalize_state(
        self,
        *,
        normalized_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        return self.state_normalizer.denormalize(value_batch=normalized_state_batch)

    def denormalize_state_displacement(
        self,
        *,
        normalized_state_displacement_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        return self.state_displacement_normalizer.denormalize(
            value_batch=normalized_state_displacement_batch
        )

    def scale_to_step_units(
        self,
        *,
        state_difference_batch: torch.Tensor,  # (..., state_dimensionality)
    ) -> torch.Tensor:  # (..., state_dimensionality)
        """A state difference measured in typical one-step motion per
        dimension. Only the scale applies — a difference needs no shift."""
        return state_difference_batch / self.state_displacement_normalizer.std
