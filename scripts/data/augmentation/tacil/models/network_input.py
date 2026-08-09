"""What TaCIL's dynamics network reads, built from a PushT state.

Two layout facts live here, both unrelated to the periodicity that
scripts/shared/pusht_state.py deals with:

- the network reads the agent's position as it is, and the block's position
  relative to the agent (network_features). Contact depends on where the block
  is relative to the agent, so that difference is handed over rather than left
  to be learned. Nothing is lost by it: the absolute block position is the sum
  of the two, so the two coordinate systems carry the same information. Keeping
  the agent's own position is what lets the model see the arena walls at all;
- two states showing the same orientation must look the same to a network, so
  the angle reaches it as a point on the unit circle, never as a number.

This is the method's own reading of the state, not a property of PushT, which
is why it sits with the models instead of with the layout.
"""

import torch

from scripts.data.augmentation.tacil.models.normalizers import Normalizers
from scripts.shared.pusht_state import (
    AGENT_POSITION_SLICE,
    ANGLE_INDEX,
    BLOCK_POSITION_SLICE,
)


def network_features(
    *,
    state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
) -> torch.Tensor:  # (batch_size, NETWORK_FEATURE_DIMENSIONALITY)
    """The quantities the network reads, before any scaling.

    The agent's position as it is, then the block and the action measured from
    the agent. Contact and the commanded motion both depend on those
    differences, so they are handed over rather than left to be worked out.

    Kept separate from network_input so the scale of each quantity can be
    measured on the quantity itself. Dividing a difference by the spread of an
    absolute position would be the wrong scale.
    """
    agent_position_batch = state_batch[:, AGENT_POSITION_SLICE]
    return torch.cat(
        tensors=[
            agent_position_batch,
            state_batch[:, BLOCK_POSITION_SLICE] - agent_position_batch,
            action_batch - agent_position_batch,
        ],
        dim=-1,
    )


# Agent position, block measured from the agent, action measured from the agent.
NETWORK_FEATURE_DIMENSIONALITY = 6
# Those, plus the orientation as a cosine and a sine.
NETWORK_INPUT_DIMENSIONALITY = NETWORK_FEATURE_DIMENSIONALITY + 2


def network_input(
    *,
    normalizers: Normalizers,
    state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
) -> torch.Tensor:  # (batch_size, NETWORK_INPUT_DIMENSIONALITY)
    """Everything the network sees: the scaled features, then the orientation.

    The orientation enters as a point on the unit circle and is left unscaled.
    Two states that show the same block orientation then look the same to the
    network, whether the angle reads 0.3 or 0.3 + 2π, and the circle stays a
    circle.
    """
    angle_batch = state_batch[:, ANGLE_INDEX : ANGLE_INDEX + 1]
    return torch.cat(
        tensors=[
            normalizers.network_feature_normalizer.normalize(
                value_batch=network_features(
                    state_batch=state_batch, action_batch=action_batch
                )
            ),
            torch.cos(angle_batch),
            torch.sin(angle_batch),
        ],
        dim=-1,
    )
