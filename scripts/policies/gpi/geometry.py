import torch

from scripts.shared.pusht_state import BLOCK_ORIENTATION_INDEX, POSITION_SLICE


def convert_position_to_block_frame(
    *,
    block_position_batch: torch.Tensor,  # (batch_size, 2)
    block_angle_batch: torch.Tensor,  # (batch_size,)
    position_batch: torch.Tensor,  # (batch_size, 2)
) -> torch.Tensor:  # (batch_size, 2)
    return _rotate(
        vector_batch=position_batch - block_position_batch,
        angle_batch=-block_angle_batch,
    )  # (batch_size, 2)


def convert_position_to_world_frame(
    *,
    block_position_batch: torch.Tensor,  # (batch_size, 2)
    block_angle_batch: torch.Tensor,  # (batch_size,)
    position_batch: torch.Tensor,  # (batch_size, 2)
) -> torch.Tensor:  # (batch_size, 2)
    rotated_position_batch = _rotate(
        vector_batch=position_batch, angle_batch=block_angle_batch
    )  # (batch_size, 2)

    return rotated_position_batch + block_position_batch  # (batch_size, 2)


def calculate_observation_distances(
    *,
    observation_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    query_observation: torch.Tensor,  # (state_dimensionality,)
) -> torch.Tensor:  # (batch_size,)
    position_difference_batch = (
        observation_batch[:, POSITION_SLICE] - query_observation[POSITION_SLICE]
    )  # (step_count, 4)

    angle_difference_batch = (
        observation_batch[:, BLOCK_ORIENTATION_INDEX]
        - query_observation[BLOCK_ORIENTATION_INDEX]
    ).abs()  # (step_count,)
    # The angle wraps at 2.0, not at 2π, because the states arrive min-max
    # normalized into [-1, 1].
    wrapped_angle_difference_batch = torch.minimum(
        input=angle_difference_batch, other=2.0 - angle_difference_batch
    )  # (step_count,)

    return torch.sqrt(
        input=(position_difference_batch**2).sum(dim=1)
        + wrapped_angle_difference_batch**2
    )  # (step_count,)


# ---- Private --------------------------------------------


def _rotate(
    *,
    vector_batch: torch.Tensor,  # (batch_size, 2)
    angle_batch: torch.Tensor,  # (batch_size,)
) -> torch.Tensor:  # (batch_size, 2)
    return torch.matmul(
        input=_create_rotation_matrix(angle_batch=angle_batch),
        other=vector_batch.unsqueeze(dim=-1),
    ).squeeze(dim=-1)


def _create_rotation_matrix(
    *,
    angle_batch: torch.Tensor,  # (batch_size,)
) -> torch.Tensor:  # (batch_size, 2, 2)
    cosine_batch = torch.cos(input=angle_batch)  # (batch_size,)
    sine_batch = torch.sin(input=angle_batch)  # (batch_size,)
    return torch.stack(
        tensors=[
            torch.stack(tensors=[cosine_batch, -sine_batch], dim=-1),
            torch.stack(tensors=[sine_batch, cosine_batch], dim=-1),
        ],
        dim=-2,
    )  # (batch_size, 2, 2)
