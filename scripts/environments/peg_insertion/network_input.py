import torch

from scripts.environments.peg_insertion.observation import (
    ACTION_DIMENSIONALITY,
    ARM_ACTION_COUNT,
    GRIPPER_OPENING_INDEX,
    HOLE_ORIENTATION_INDEX,
    HOLE_POSITION_SLICE,
    HOLE_RADIUS_INDEX,
    PEG_ANGULAR_VELOCITY_SLICE,
    PEG_HALF_SIZE_SLICE,
    PEG_LINEAR_VELOCITY_SLICE,
    PEG_ORIENTATION_SLICE,
    PEG_POSITION_SLICE,
    TCP_ANGULAR_VELOCITY_SLICE,
    TCP_LINEAR_VELOCITY_SLICE,
    TCP_ORIENTATION_SLICE,
    TCP_POSITION_SLICE,
)

_ARM_ACTION_SLICE = slice(0, ARM_ACTION_COUNT)
_GRIPPER_ACTION_SLICE = slice(ARM_ACTION_COUNT, ACTION_DIMENSIONALITY)


def build_network_input(
    *,
    state_batch: torch.Tensor,  # (batch_size, STATE_DIMENSIONALITY)
    action_batch: torch.Tensor,  # (batch_size, ACTION_DIMENSIONALITY)
) -> torch.Tensor:  # (batch_size, network_input_dimensionality)
    tcp_position_batch = state_batch[..., TCP_POSITION_SLICE]  # (batch_size, 3)
    peg_position_batch = state_batch[..., PEG_POSITION_SLICE]  # (batch_size, 3)
    hole_orientation_batch = state_batch[
        ..., HOLE_ORIENTATION_INDEX : HOLE_ORIENTATION_INDEX + 1
    ]  # (batch_size, 1)

    return torch.cat(
        tensors=[
            tcp_position_batch,
            state_batch[..., TCP_ORIENTATION_SLICE],
            state_batch[..., TCP_LINEAR_VELOCITY_SLICE],
            state_batch[..., TCP_ANGULAR_VELOCITY_SLICE],
            state_batch[..., GRIPPER_OPENING_INDEX : GRIPPER_OPENING_INDEX + 1],
            peg_position_batch - tcp_position_batch,
            state_batch[..., PEG_ORIENTATION_SLICE],
            state_batch[..., PEG_LINEAR_VELOCITY_SLICE],
            state_batch[..., PEG_ANGULAR_VELOCITY_SLICE],
            state_batch[..., HOLE_POSITION_SLICE] - peg_position_batch,
            torch.sin(hole_orientation_batch),
            torch.cos(hole_orientation_batch),
            state_batch[..., PEG_HALF_SIZE_SLICE],
            state_batch[..., HOLE_RADIUS_INDEX : HOLE_RADIUS_INDEX + 1],
            action_batch[..., _ARM_ACTION_SLICE],
            action_batch[..., _GRIPPER_ACTION_SLICE],
        ],
        dim=-1,
    )
