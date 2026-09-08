import torch

from scripts.environments.pusht.state_space import (
    AGENT_POSITION_SLICE,
    BLOCK_ORIENTATION_INDEX,
    BLOCK_POSITION_SLICE,
)
from scripts.policies.gpi.geometry import (
    calculate_observation_distances,
    convert_position_to_block_frame,
    convert_position_to_world_frame,
)


class PushTGpiGeometry:
    """The agent is a point on a plane, and the object frame is the block's
    own pose."""

    @property
    def agent_position_slice(self) -> slice:
        return AGENT_POSITION_SLICE

    @property
    def action_position_slice(self) -> slice:
        # An action is the agent's target position and nothing else.
        return slice(0, AGENT_POSITION_SLICE.stop - AGENT_POSITION_SLICE.start)

    def convert_actions_to_object_frame(
        self,
        *,
        observation_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, 2)
    ) -> torch.Tensor:  # (batch_size, 2)
        return convert_position_to_block_frame(
            block_position_batch=observation_batch[:, BLOCK_POSITION_SLICE],
            block_angle_batch=observation_batch[:, BLOCK_ORIENTATION_INDEX],
            position_batch=action_batch,
        )

    def convert_actions_to_world_frame(
        self,
        *,
        observation_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, 2)
    ) -> torch.Tensor:  # (batch_size, 2)
        return convert_position_to_world_frame(
            block_position_batch=observation_batch[:, BLOCK_POSITION_SLICE],
            block_angle_batch=observation_batch[:, BLOCK_ORIENTATION_INDEX],
            position_batch=action_batch,
        )

    def calculate_observation_distances(
        self,
        *,
        observation_batch: torch.Tensor,  # (step_count, state_dimensionality)
        query_observation: torch.Tensor,  # (state_dimensionality,)
    ) -> torch.Tensor:  # (step_count,)
        return calculate_observation_distances(
            observation_batch=observation_batch, query_observation=query_observation
        )
