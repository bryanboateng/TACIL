import torch

from scripts.environments.pusht.observation import (
    AGENT_POSITION_SLICE,
    BLOCK_ORIENTATION_INDEX,
    BLOCK_POSITION_SLICE,
)


def build_network_input(
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
