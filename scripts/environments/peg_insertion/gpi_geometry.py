import torch

from scripts.environments.peg_insertion.observation import (
    ARM_JOINT_COUNT,
    JOINT_ANGLE_SLICE,
)

_AGENT_POSITION_SLICE = slice(JOINT_ANGLE_SLICE.start, ARM_JOINT_COUNT)
_ACTION_POSITION_SLICE = slice(0, ARM_JOINT_COUNT)


class PegInsertionGpiGeometry:
    @property
    def agent_position_slice(self) -> slice:
        return _AGENT_POSITION_SLICE

    @property
    def action_position_slice(self) -> slice:
        return _ACTION_POSITION_SLICE

    def convert_actions_to_object_frame(
        self,
        *,
        observation_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, action_dimensionality)
        del observation_batch
        return action_batch

    def convert_actions_to_world_frame(
        self,
        *,
        observation_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, action_dimensionality)
        del observation_batch
        return action_batch

    def calculate_observation_distances(
        self,
        *,
        observation_batch: torch.Tensor,  # (step_count, state_dimensionality)
        query_observation: torch.Tensor,  # (state_dimensionality,)
    ) -> torch.Tensor:  # (step_count,)
        return torch.linalg.norm(observation_batch - query_observation, dim=1)
