import torch

from scripts.environments.peg_insertion.observation import (
    ACTION_DIMENSIONALITY,
    ARM_JOINT_COUNT,
    GRIPPER_POSITION_SLICE,
    HOLE_ORIENTATION_INDEX,
    HOLE_POSITION_SLICE,
    HOLE_RADIUS_INDEX,
    JOINT_ANGLE_SLICE,
    JOINT_VELOCITY_SLICE,
    PEG_HALF_SIZE_SLICE,
    PEG_ORIENTATION_INDEX,
    PEG_POSITION_SLICE,
)

_ARM_JOINT_SLICE = slice(0, ARM_JOINT_COUNT)
_ARM_ACTION_SLICE = slice(0, ARM_JOINT_COUNT)
_GRIPPER_ACTION_SLICE = slice(ARM_JOINT_COUNT, ACTION_DIMENSIONALITY)


def build_network_input(
    *,
    state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
) -> torch.Tensor:  # (batch_size, network_input_dimensionality)
    gripper_position_batch = state_batch[..., GRIPPER_POSITION_SLICE]  # (batch_size, 3)
    peg_position_batch = state_batch[..., PEG_POSITION_SLICE]  # (batch_size, 3)
    peg_orientation_batch = state_batch[
        ..., PEG_ORIENTATION_INDEX : PEG_ORIENTATION_INDEX + 1
    ]  # (batch_size, 1)
    hole_orientation_batch = state_batch[
        ..., HOLE_ORIENTATION_INDEX : HOLE_ORIENTATION_INDEX + 1
    ]  # (batch_size, 1)

    return torch.cat(
        tensors=[
            state_batch[..., JOINT_ANGLE_SLICE],
            state_batch[..., JOINT_VELOCITY_SLICE],
            gripper_position_batch,
            peg_position_batch - gripper_position_batch,
            state_batch[..., HOLE_POSITION_SLICE] - peg_position_batch,
            torch.sin(peg_orientation_batch),
            torch.cos(peg_orientation_batch),
            torch.sin(hole_orientation_batch),
            torch.cos(hole_orientation_batch),
            state_batch[..., PEG_HALF_SIZE_SLICE],
            state_batch[..., HOLE_RADIUS_INDEX : HOLE_RADIUS_INDEX + 1],
            action_batch[..., _ARM_ACTION_SLICE] - state_batch[..., _ARM_JOINT_SLICE],
            action_batch[..., _GRIPPER_ACTION_SLICE],
        ],
        dim=-1,
    )
