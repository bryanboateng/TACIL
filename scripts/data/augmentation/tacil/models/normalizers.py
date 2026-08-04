from typing import Self

import torch

from scripts.shared.normalization import ZScoreNormalizer

# PushT frame layout: agent x/y, block x/y, block angle. The dynamics state is
# one such frame, or two when the previous state is stacked in.
_FRAME_SIZE = 5
_AGENT_SLICE = slice(0, 2)
_BLOCK_SLICE = slice(2, 4)
_ANGLE_INDEX = 4


def agent_relative_input_dimensionality(
    *, state_dimensionality: int, action_dimensionality: int
) -> int:
    """Same as state plus action, minus the agent's own position: measured
    from the agent, that position is always zero."""
    return state_dimensionality - 2 + action_dimensionality


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

    def normalized_input(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality + action_dimensionality)
        return torch.cat(
            tensors=[
                self.state_normalizer.normalize(value_batch=state_batch),
                self.action_normalizer.normalize(value_batch=action_batch),
            ],
            dim=-1,
        )

    def agent_relative_input(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, agent_relative_input_dimensionality)
        """The network input measured from the agent's current position.

        Contact depends on where things are relative to the agent, not on
        where in the arena that happens. Feeding differences instead of
        absolute coordinates lets the model learn the contact rule once
        instead of once per place. Scales stay the plain per-quantity ones,
        so the inputs keep a sane magnitude.
        """
        agent_position_batch = state_batch[:, _AGENT_SLICE]
        parts = [
            (action_batch - agent_position_batch) / self.action_normalizer.std
        ]

        for frame_start in range(0, state_batch.shape[-1], _FRAME_SIZE):
            frame_batch = state_batch[:, frame_start : frame_start + _FRAME_SIZE]
            frame_scale = self.state_normalizer.std[
                frame_start : frame_start + _FRAME_SIZE
            ]
            if frame_start > 0:
                # The current agent position is the origin, so only earlier
                # frames still carry one.
                parts.append(
                    (frame_batch[:, _AGENT_SLICE] - agent_position_batch)
                    / frame_scale[_AGENT_SLICE]
                )
            parts.append(
                (frame_batch[:, _BLOCK_SLICE] - agent_position_batch)
                / frame_scale[_BLOCK_SLICE]
            )
            # The angle needs no origin; it is normalized as usual.
            parts.append(
                (
                    frame_batch[:, _ANGLE_INDEX : _ANGLE_INDEX + 1]
                    - self.state_normalizer.mean[frame_start + _ANGLE_INDEX]
                )
                / frame_scale[_ANGLE_INDEX]
            )

        return torch.cat(tensors=parts, dim=-1)

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
