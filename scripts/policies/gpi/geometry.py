"""GPI's geometry, which reads the state as a PushT state.

The method needs more than a width: it needs one rigid object with a pose and
actions that are agent target positions in the object's plane. The layout comes
from scripts/shared/pusht_state.py so this dependency shows up in the import
graph rather than hiding in slice literals.

Two coordinate systems meet here, so each function names the one it works in:

- se2_to_relative_action and relative_action_to_global take raw states and raw
  actions, in the environment's pixels;
- distance takes min-max normalized states, because the database indexes the
  normalized demonstrations.
"""

import torch

from scripts.shared.pusht_state import ANGLE_INDEX, BLOCK_POSITION_SLICE, POSITION_SLICE


def se2_to_relative_action(
    *,
    states: torch.Tensor,  # (N, state_dimensionality), raw
    actions: torch.Tensor,  # (N, action_dimensionality), raw agent target
) -> torch.Tensor:  # (N, action_dimensionality) in the object's local SE(2) frame
    """Convert global agent targets into the object's local SE(2) frame."""
    object_position = states[:, BLOCK_POSITION_SLICE]  # (N, 2)
    object_angle = states[:, ANGLE_INDEX]  # (N,)
    cosine = torch.cos(input=object_angle)  # (N,)
    sine = torch.sin(input=object_angle)  # (N,)
    rotation = torch.stack(
        tensors=[
            torch.stack(tensors=[cosine, -sine], dim=-1),
            torch.stack(tensors=[sine, cosine], dim=-1),
        ],
        dim=-1,
    )  # (N, 2, 2)
    translated = actions - object_position  # (N, 2)
    return torch.bmm(input=rotation, mat2=translated.unsqueeze(dim=-1)).squeeze(
        dim=-1
    )  # (N, 2)


def relative_action_to_global(
    *,
    observation: torch.Tensor,  # (batch_size, state_dimensionality), raw
    action: torch.Tensor,  # (batch_size, action_dimensionality) in the object's frame
) -> torch.Tensor:  # (batch_size, action_dimensionality), raw world coordinates
    """Inverse of ``se2_to_relative_action``: object-centric -> world coords."""
    object_position = observation[:, BLOCK_POSITION_SLICE]  # (batch_size, 2)
    object_angle = observation[:, ANGLE_INDEX]  # (batch_size,)
    cosine = torch.cos(input=object_angle)  # (batch_size,)
    sine = torch.sin(input=object_angle)  # (batch_size,)
    inverse_rotation = torch.stack(
        tensors=[
            torch.stack(tensors=[cosine, sine], dim=-1),
            torch.stack(tensors=[-sine, cosine], dim=-1),
        ],
        dim=-1,
    )  # (batch_size, 2, 2)
    rotated = torch.bmm(input=inverse_rotation, mat2=action.unsqueeze(dim=-1)).squeeze(
        dim=-1
    )  # (batch_size, 2)
    return rotated + object_position  # (batch_size, 2)


def distance(
    *,
    states: torch.Tensor,  # (N, state_dimensionality), min-max normalized
    query: torch.Tensor,  # (1, state_dimensionality), min-max normalized
) -> torch.Tensor:  # (N,)
    """Euclidean distance over the positions plus the wrapped angle.

    The angle wraps at 2.0, not at 2π, because the states arrive min-max
    normalized into [-1, 1]. That treats the recorded angle range as one full
    turn, which is exact only when the demonstrations cover the whole circle.
    Left as it is: this is the baseline's own distance, not TaCIL's ⊟.
    """
    position_difference = states[:, POSITION_SLICE] - query[:, POSITION_SLICE]  # (N, 4)
    position_distance_squared = torch.sum(input=position_difference**2, dim=1)  # (N,)
    angle_difference = states[:, ANGLE_INDEX] - query[:, ANGLE_INDEX]  # (N,)
    absolute_angle = torch.abs(input=angle_difference)  # (N,)
    wrapped_angle = torch.min(input=absolute_angle, other=2.0 - absolute_angle)  # (N,)
    return torch.sqrt(input=position_distance_squared + wrapped_angle**2)  # (N,)
