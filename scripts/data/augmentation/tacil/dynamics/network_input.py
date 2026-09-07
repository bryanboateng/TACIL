import torch

from scripts.shared.pusht_state import (
    AGENT_POSITION_SLICE,
    BLOCK_ORIENTATION_INDEX,
    BLOCK_POSITION_SLICE,
)

# The agent's absolute position
_AGENT_POSITION_DIMENSIONALITY = AGENT_POSITION_SLICE.stop - AGENT_POSITION_SLICE.start

# The orientation of the block as sine and cosine
_BLOCK_ORIENTATION_DIMENSIONALITY = 2

# The block position relative to the agent
_RELATIVE_BLOCK_POSITION_DIMENSIONALITY = (
    BLOCK_POSITION_SLICE.stop - BLOCK_POSITION_SLICE.start
)

# The action relative to the agent
_RELATIVE_ACTION_DIMENSIONALITY = AGENT_POSITION_SLICE.stop - AGENT_POSITION_SLICE.start


NETWORK_INPUT_DIMENSIONALITY = (
    _AGENT_POSITION_DIMENSIONALITY
    + _BLOCK_ORIENTATION_DIMENSIONALITY
    + _RELATIVE_BLOCK_POSITION_DIMENSIONALITY
    + _RELATIVE_ACTION_DIMENSIONALITY
)


def create_raw_network_input(
    *, state_batch: torch.Tensor, action_batch: torch.Tensor
) -> torch.Tensor:  # (batch_size, NETWORK_INPUT_DIMENSIONALITY)
    block_orientation_batch = state_batch[
        :, BLOCK_ORIENTATION_INDEX : BLOCK_ORIENTATION_INDEX + 1
    ]
    agent_position_batch = state_batch[:, AGENT_POSITION_SLICE]
    return torch.cat(
        tensors=[
            # Absolute agent position
            state_batch[:, AGENT_POSITION_SLICE],
            # Block orientation sine
            torch.sin(block_orientation_batch),
            # Block orientation cosine
            torch.cos(block_orientation_batch),
            # Block position relative to the agent
            state_batch[:, BLOCK_POSITION_SLICE] - agent_position_batch,
            # Action relative to the agent
            action_batch - agent_position_batch,
        ],
        dim=-1,
    )
