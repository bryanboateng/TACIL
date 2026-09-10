import torch

from scripts.environments.maniskill.observation import (
    POSE_ACTION_POSITION_SLICE,
    agent_position_slice,
)


class ManiSkillGpiGeometry:
    """The agent is the gripper, and actions are already written in the world
    frame.

    There is no object frame here. The reference makes that frame optional
    through its `use_relative_action` switch (GPI/gpi/policies/base.py), and
    the cube barely turns while it is pushed: 0.00 degrees per step at the
    median and 1.55 at the 99th percentile. A frame that never rotates adds
    nothing, so both conversions hand the action straight back.
    """

    def __init__(self, *, task_id: str) -> None:
        self._agent_position_slice = agent_position_slice(task_id=task_id)

    @property
    def agent_position_slice(self) -> slice:
        return self._agent_position_slice

    @property
    def action_position_slice(self) -> slice:
        return POSE_ACTION_POSITION_SLICE

    def convert_actions_to_object_frame(
        self,
        *,
        observation_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, action_dimensionality)
        return action_batch

    def convert_actions_to_world_frame(
        self,
        *,
        observation_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, action_dimensionality)
        return action_batch

    def calculate_observation_distances(
        self,
        *,
        observation_batch: torch.Tensor,  # (step_count, state_dimensionality)
        query_observation: torch.Tensor,  # (state_dimensionality,)
    ) -> torch.Tensor:  # (step_count,)
        """Euclidean distance over the whole state.

        Every entry counts, as in the reference, whose PushT distance also
        spans its whole five entry state. No entry needs wrapping here,
        because the rotations were dropped from the state.
        """
        return torch.linalg.norm(
            observation_batch - query_observation, dim=1
        )  # (step_count,)
