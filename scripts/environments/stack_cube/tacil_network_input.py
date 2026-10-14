import torch

from scripts.environments.stack_cube.observation import (
    CUBE_A_POSITION_SLICE,
    CUBE_B_POSITION_SLICE,
    TCP_POSITION_SLICE,
)


def build_network_input(
    *,
    state_batch: torch.Tensor,  # (batch_size, STATE_DIMENSIONALITY)
    action_batch: torch.Tensor,  # (batch_size, ACTION_DIMENSIONALITY)
) -> torch.Tensor:  # (batch_size, network_input_dimensionality)
    relative_state_batch = state_batch.clone()
    relative_state_batch[..., CUBE_A_POSITION_SLICE] = (
        state_batch[..., CUBE_A_POSITION_SLICE] - state_batch[..., TCP_POSITION_SLICE]
    )
    relative_state_batch[..., CUBE_B_POSITION_SLICE] = (
        state_batch[..., CUBE_B_POSITION_SLICE]
        - state_batch[..., CUBE_A_POSITION_SLICE]
    )
    return torch.cat(tensors=[relative_state_batch, action_batch], dim=-1)
