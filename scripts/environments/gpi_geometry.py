from typing import Protocol

import torch


class GpiGeometry(Protocol):
    """What GPI needs to know about one environment.

    GPI reads a state, finds the nearest recorded state, and steers the agent
    along that recording. Three things about that are environment specific:
    where the agent sits inside a state, how near two states are, and which
    frame an action is written in. The reference implementation factors
    exactly these out as well, into its `EpisodeDataset` protocol
    (GPI/gpi/types.py).
    """

    @property
    def agent_position_slice(self) -> slice:
        """Where the position of the thing the policy moves sits in a state."""
        ...

    @property
    def action_position_slice(self) -> slice:
        """Where that same position is named inside an action.

        GPI subtracts the one from the other, so both must count the same
        number of entries and be measured in the same frame. Anything else the
        action carries, such as a gripper opening, GPI leaves alone. In PushT
        the two slices happen to coincide, which is why the reference gets
        away with writing `[:2]` for both.
        """
        ...

    def convert_actions_to_object_frame(
        self,
        *,
        observation_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, action_dimensionality)
        """Rewrite actions relative to the object being manipulated.

        An environment that does not use an object frame returns the actions
        unchanged, which is what the reference does when `use_relative_action`
        is off (GPI/gpi/policies/base.py).
        """
        ...

    def convert_actions_to_world_frame(
        self,
        *,
        observation_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, action_dimensionality)
        """The inverse of `convert_actions_to_object_frame`."""
        ...

    def calculate_observation_distances(
        self,
        *,
        observation_batch: torch.Tensor,  # (step_count, state_dimensionality)
        query_observation: torch.Tensor,  # (state_dimensionality,)
    ) -> torch.Tensor:  # (step_count,)
        """How near each recorded state is to the queried one."""
        ...
