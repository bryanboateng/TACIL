import torch

from scripts.data.augmentation.tacil.models.normalizers import Normalizers
from scripts.shared.pusht_state import (
    AGENT_POSITION_SLICE,
    BLOCK_ANGLE_INDEX,
    BLOCK_POSITION_SLICE,
)


def relative_displacements(
    *,
    state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
) -> torch.Tensor:  # (batch_size, RELATIVE_DISPLACEMENT_DIMENSIONALITY)
    agent_position_batch = state_batch[:, AGENT_POSITION_SLICE]
    return torch.cat(
        tensors=[
            state_batch[:, BLOCK_POSITION_SLICE] - agent_position_batch,
            action_batch - agent_position_batch,
        ],
        dim=-1,
    )


# Block measured from the agent, action measured from the agent: the contact
# geometry that drives the physics, handed to the network directly instead of
# left for it to infer from absolute coordinates.
RELATIVE_DISPLACEMENT_DIMENSIONALITY = (
    BLOCK_POSITION_SLICE.stop - BLOCK_POSITION_SLICE.start
) + (AGENT_POSITION_SLICE.stop - AGENT_POSITION_SLICE.start)
# The orientation as a cosine and a sine.
_ORIENTATION_DIMENSIONALITY = 2
# The agent's absolute position: the arena has walls, so where the agent sits
# in the arena is itself information the model needs, not something to
# normalize away.
_AGENT_POSITION_DIMENSIONALITY = AGENT_POSITION_SLICE.stop - AGENT_POSITION_SLICE.start

NETWORK_INPUT_DIMENSIONALITY = (
    RELATIVE_DISPLACEMENT_DIMENSIONALITY
    + _ORIENTATION_DIMENSIONALITY
    + _AGENT_POSITION_DIMENSIONALITY
)


def raw_network_input(
    *, state_batch: torch.Tensor, action_batch: torch.Tensor
) -> torch.Tensor:  # (batch_size, NETWORK_INPUT_DIMENSIONALITY)
    """Everything the network sees, before scaling: the contact geometry
    (block and action, both measured from the agent), the orientation as a
    point on the unit circle, then the agent's absolute position.

    The contact geometry is handed to the network directly rather than left
    for it to infer from absolute coordinates, since it is what drives the
    push. The absolute agent position is added on top: the arena has walls,
    so where the agent sits is itself information the model needs.
    """
    angle_batch = state_batch[:, BLOCK_ANGLE_INDEX : BLOCK_ANGLE_INDEX + 1]
    return torch.cat(
        tensors=[
            relative_displacements(state_batch=state_batch, action_batch=action_batch),
            torch.cos(angle_batch),
            torch.sin(angle_batch),
            state_batch[:, AGENT_POSITION_SLICE],
        ],
        dim=-1,
    )


def network_input(
    *, state_batch: torch.Tensor, action_batch: torch.Tensor, normalizers: Normalizers
) -> torch.Tensor:  # (batch_size, NETWORK_INPUT_DIMENSIONALITY)
    """The scaled version of raw_network_input: one normalizer, fit per
    dimension on the whole vector, applied once."""
    return normalizers.network_input_normalizer.normalize(
        value_batch=raw_network_input(state_batch=state_batch, action_batch=action_batch)
    )
