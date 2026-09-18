import torch

from scripts.environments.pusht.state_space import (
    ACTION_DIMENSIONALITY,
    AGENT_POSITION_SLICE,
    BLOCK_ORIENTATION_INDEX,
    BLOCK_POSITION_SLICE,
    POSITION_SLICE,
)

_ACTION_POSITION_SLICE = slice(0, ACTION_DIMENSIONALITY)


class PushTGpiGeometry:
    @property
    def agent_position_slice(self) -> slice:
        return AGENT_POSITION_SLICE

    @property
    def action_position_slice(self) -> slice:
        return _ACTION_POSITION_SLICE

    def convert_actions_to_object_frame(
        self,
        *,
        observation_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, action_dimensionality)
        return _rotate(
            vector_batch=action_batch
            - observation_batch[:, BLOCK_POSITION_SLICE],  # (batch_size, 2)
            angle_batch=-observation_batch[:, BLOCK_ORIENTATION_INDEX],
        )

    def convert_actions_to_world_frame(
        self,
        *,
        observation_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, action_dimensionality)
        return (
            _rotate(
                vector_batch=action_batch,
                angle_batch=observation_batch[:, BLOCK_ORIENTATION_INDEX],
            )
            + observation_batch[:, BLOCK_POSITION_SLICE]
        )

    def calculate_observation_distances(
        self,
        *,
        observation_batch: torch.Tensor,  # (step_count, state_dimensionality)
        query_observation: torch.Tensor,  # (state_dimensionality,)
    ) -> torch.Tensor:  # (step_count,)
        position_difference_batch = (
            observation_batch[:, POSITION_SLICE] - query_observation[POSITION_SLICE]
        )  # (step_count, 4)

        angle_difference_batch = (
            observation_batch[:, BLOCK_ORIENTATION_INDEX]
            - query_observation[BLOCK_ORIENTATION_INDEX]
        ).abs()  # (step_count,)
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
