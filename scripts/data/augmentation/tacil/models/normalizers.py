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
        # Taken by the caller rather than computed here: subtracting two states
        # is the state layout's business, and this class knows only scales.
        state_displacement_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    ) -> Self:
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

    def scale_to_state_spread_units(
        self,
        *,
        state_difference_batch: torch.Tensor,  # (..., state_dimensionality)
    ) -> torch.Tensor:  # (..., state_dimensionality)
        """A state difference measured in the spread of the whole state per
        dimension — the scale normalize_state works in.

        This is the space distances between two states are compared in: the
        error budget, the inverse dynamics residual and the ensemble's
        disagreement. Only the scale applies, since the shift cancels in a
        difference.
        """
        return state_difference_batch / self.state_normalizer.std

    def unscale_from_state_spread_units(
        self,
        *,
        scaled_state_difference_batch: torch.Tensor,  # (..., state_dimensionality)
    ) -> torch.Tensor:  # (..., state_dimensionality)
        """The inverse of scale_to_state_spread_units: turns a step measured in
        the normalized state space back into one the state itself can take."""
        return scaled_state_difference_batch * self.state_normalizer.std
