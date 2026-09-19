import torch

from scripts.environments.network_input import NetworkInput
from scripts.environments.pusht.state_space import (
    ACTION_DIMENSIONALITY,
    AGENT_POSITION_SLICE,
    BLOCK_ORIENTATION_INDEX,
    BLOCK_POSITION_SLICE,
)

_AGENT_POSITION_DIMENSIONALITY = AGENT_POSITION_SLICE.stop - AGENT_POSITION_SLICE.start
_BLOCK_ORIENTATION_DIMENSIONALITY = 2
_RELATIVE_BLOCK_POSITION_DIMENSIONALITY = (
    BLOCK_POSITION_SLICE.stop - BLOCK_POSITION_SLICE.start
)
_RELATIVE_ACTION_DIMENSIONALITY = ACTION_DIMENSIONALITY

_DIMENSIONALITY = (
    _AGENT_POSITION_DIMENSIONALITY
    + _BLOCK_ORIENTATION_DIMENSIONALITY
    + _RELATIVE_BLOCK_POSITION_DIMENSIONALITY
    + _RELATIVE_ACTION_DIMENSIONALITY
)


def create_network_input() -> NetworkInput:
    return NetworkInput(dimensionality=_DIMENSIONALITY, builder=_build)


# ---- Private --------------------------------------------


def _build(
    *,
    state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
) -> torch.Tensor:  # (batch_size, network_input_dimensionality)
    block_orientation_batch = state_batch[
        ..., BLOCK_ORIENTATION_INDEX : BLOCK_ORIENTATION_INDEX + 1
    ]  # (batch_size, 1)
    agent_position_batch = state_batch[..., AGENT_POSITION_SLICE]  # (batch_size, 2)
    return torch.cat(
        tensors=[
            agent_position_batch,
            torch.sin(block_orientation_batch),
            torch.cos(block_orientation_batch),
            state_batch[..., BLOCK_POSITION_SLICE] - agent_position_batch,
            action_batch - agent_position_batch,
        ],
        dim=-1,
    )
