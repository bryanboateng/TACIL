import torch
from mani_skill.utils.geometry.rotation_conversions import rotation_6d_to_matrix

ARM_ACTION_COUNT = 6
ACTION_DIMENSIONALITY = 7

ACTION_TRANSLATION_SLICE = slice(0, 3)
ACTION_ROTATION_SLICE = slice(3, ARM_ACTION_COUNT)
ACTION_GRIPPER_SLICE = slice(ARM_ACTION_COUNT, ARM_ACTION_COUNT + 1)

TRANSLATION_SCALE_IN_METRES = 0.1
ROTATION_SCALE_IN_RADIANS = -0.1
ROTATION_MAGNITUDE_IN_RADIANS = abs(ROTATION_SCALE_IN_RADIANS)
GRIPPER_RANGE_IN_METRES = 0.05


def read_rotations(
    *,
    observation_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    orientation_slice: slice,
) -> torch.Tensor:  # (batch_size, 3, 3)
    return rotation_6d_to_matrix(d6=observation_batch[:, orientation_slice]).transpose(
        dim0=-2, dim1=-1
    )


def rotate_into_frame(
    *,
    vector_batch: torch.Tensor,  # (batch_size, 3)
    frame_rotation: torch.Tensor,  # (batch_size, 3, 3)
) -> torch.Tensor:  # (batch_size, 3)
    return torch.matmul(
        input=frame_rotation.transpose(dim0=-2, dim1=-1),
        other=vector_batch.unsqueeze(dim=-1),
    ).squeeze(dim=-1)


def flatten_rotations_in_frame(
    *,
    rotation_batch: torch.Tensor,  # (batch_size, 3, 3)
    frame_rotation: torch.Tensor,  # (batch_size, 3, 3)
) -> torch.Tensor:  # (batch_size, 9)
    return (frame_rotation.transpose(dim0=-2, dim1=-1) @ rotation_batch).flatten(
        start_dim=-2
    )


def clip_to_unit_norm(
    *,
    vector_batch: torch.Tensor,  # (batch_size, 3)
) -> torch.Tensor:  # (batch_size, 3)
    norm = torch.linalg.norm(vector_batch, dim=-1, keepdim=True)  # (batch_size, 1)
    return torch.where(
        condition=norm > 1.0, input=vector_batch / norm, other=vector_batch
    )


def calculate_norms(
    *,
    difference_batch: torch.Tensor,  # (step_count, feature_dimensionality)
    feature_slice: slice,
) -> torch.Tensor:  # (step_count,)
    return torch.linalg.norm(difference_batch[:, feature_slice], dim=-1)


def calculate_rotation_angles(
    *,
    feature_batch: torch.Tensor,  # (step_count, feature_dimensionality)
    query_feature: torch.Tensor,  # (feature_dimensionality,)
    rotation_slice: slice,
) -> torch.Tensor:  # (step_count,)
    trace_batch = (
        feature_batch[:, rotation_slice] * query_feature[rotation_slice]
    ).sum(dim=-1)  # (step_count,)
    return torch.arccos(input=((trace_batch - 1.0) / 2.0).clamp(min=-1.0, max=1.0))
