import torch


def se2_to_relative_action(
    states: torch.Tensor, actions: torch.Tensor
) -> torch.Tensor:
    """Convert global agent targets into the object's local SE(2) frame."""
    object_xy = states[:, 2:4]
    object_theta = states[:, 4]
    cos_theta = torch.cos(input=object_theta)
    sin_theta = torch.sin(input=object_theta)
    rotation = torch.stack(
        tensors=[
            torch.stack(tensors=[cos_theta, -sin_theta], dim=-1),
            torch.stack(tensors=[sin_theta, cos_theta], dim=-1),
        ],
        dim=-1,
    )
    translated = actions[:, :2] - object_xy
    return torch.bmm(
        input=rotation, mat2=translated.unsqueeze(dim=-1)
    ).squeeze(dim=-1)


def relative_action_to_global(
    observation: torch.Tensor, action: torch.Tensor
) -> torch.Tensor:
    """Inverse of ``se2_to_relative_action``: object-centric -> world coords."""
    object_xy = observation[:, 2:4]
    object_theta = observation[:, 4]
    cos_theta = torch.cos(input=object_theta)
    sin_theta = torch.sin(input=object_theta)
    rotation_inv = torch.stack(
        tensors=[
            torch.stack(tensors=[cos_theta, sin_theta], dim=-1),
            torch.stack(tensors=[-sin_theta, cos_theta], dim=-1),
        ],
        dim=-1,
    )
    rotated = torch.bmm(
        input=rotation_inv, mat2=action[:, :2].unsqueeze(dim=-1)
    ).squeeze(dim=-1)
    return rotated + object_xy


def distance(states: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """Euclidean distance in ``(x, y, obj_x, obj_y)`` plus wrapped angle."""
    position_difference = states[:, :4] - query[:, :4]
    position_distance_squared = torch.sum(input=position_difference**2, dim=1)
    angle_difference = states[:, 4] - query[:, 4]
    absolute_angle = torch.abs(input=angle_difference)
    wrapped_angle = torch.min(input=absolute_angle, other=2.0 - absolute_angle)
    return torch.sqrt(input=position_distance_squared + wrapped_angle**2)
