import torch

AGENT_POSITION_SLICE = slice(0, 2)
BLOCK_POSITION_SLICE = slice(2, 4)
POSITION_SLICE = slice(0, 4)
BLOCK_ANGLE_INDEX = 4

STATE_GROUPS = (
    ("agent", AGENT_POSITION_SLICE),
    ("block", BLOCK_POSITION_SLICE),
    ("angle", slice(BLOCK_ANGLE_INDEX, BLOCK_ANGLE_INDEX + 1)),
)

ARENA_SIZE = 512

FULL_TURN = 2.0 * torch.pi


def subtract_states(
    *,
    minuend_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    subtrahend_batch: torch.Tensor,  # (batch_size, state_dimensionality)
) -> torch.Tensor:  # (batch_size, state_dimensionality)
    difference_batch = minuend_batch - subtrahend_batch
    difference_batch[..., BLOCK_ANGLE_INDEX] = _wrap_angle_to_pi_range(
        angle_batch=difference_batch[..., BLOCK_ANGLE_INDEX]
    )
    return difference_batch


def add_states(
    *,
    first_addend_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    second_addend_batch: torch.Tensor,  # (batch_size, state_dimensionality)
) -> torch.Tensor:  # (batch_size, state_dimensionality)
    sum_batch = first_addend_batch + second_addend_batch

    # Not `%=`: vmap has no batching rule for the in-place remainder and
    # falls back to a loop, which costs more.
    sum_batch[..., BLOCK_ANGLE_INDEX] = sum_batch[..., BLOCK_ANGLE_INDEX] % FULL_TURN

    return sum_batch


def average_states(
    *,
    state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
) -> torch.Tensor:  # (state_dimensionality,)
    average_batch = state_batch.mean(dim=0)

    angle_batch = state_batch[..., BLOCK_ANGLE_INDEX]
    average_batch[..., BLOCK_ANGLE_INDEX] = (
        torch.atan2(
            input=torch.sin(input=angle_batch).mean(dim=0),
            other=torch.cos(input=angle_batch).mean(dim=0),
        )
        % FULL_TURN
    )
    return average_batch


# ---- Private --------------------------------------------


def _wrap_angle_to_pi_range(
    *,
    angle_batch: torch.Tensor,  # (batch_size,)
) -> torch.Tensor:  # (batch_size,)
    """Wrap angles to [-π, π)."""
    return (angle_batch + torch.pi) % FULL_TURN - torch.pi
