import torch


def se2_to_relative_action(
    *,
    states: torch.Tensor,  # (N, 5) — (agent_x, agent_y, object_x, object_y, object_theta)
    actions: torch.Tensor,  # (N, 2) — global agent target (x, y)
) -> torch.Tensor:  # (N, 2) — relative action in object's local SE(2) frame
    """Convert global agent targets into the object's local SE(2) frame."""
    object_xy = states[:, 2:4]  # (N, 2)
    object_theta = states[:, 4]  # (N,)
    cos_theta = torch.cos(input=object_theta)  # (N,)
    sin_theta = torch.sin(input=object_theta)  # (N,)
    rotation = torch.stack(
        tensors=[
            torch.stack(tensors=[cos_theta, -sin_theta], dim=-1),
            torch.stack(tensors=[sin_theta, cos_theta], dim=-1),
        ],
        dim=-1,
    )  # (N, 2, 2)
    translated = actions[:, :2] - object_xy  # (N, 2)
    return torch.bmm(input=rotation, mat2=translated.unsqueeze(dim=-1)).squeeze(
        dim=-1
    )  # (N, 2)


def relative_action_to_global(
    *,
    observation: torch.Tensor,  # (batch_size, 5) — (agent_x, agent_y, object_x, object_y, object_theta)
    action: torch.Tensor,  # (batch_size, 2) — action in object-centric frame
) -> torch.Tensor:  # (batch_size, 2) — global world coordinates
    """Inverse of ``se2_to_relative_action``: object-centric -> world coords."""
    object_xy = observation[:, 2:4]  # (batch_size, 2)
    object_theta = observation[:, 4]  # (batch_size,)
    cos_theta = torch.cos(input=object_theta)  # (batch_size,)
    sin_theta = torch.sin(input=object_theta)  # (batch_size,)
    rotation_inv = torch.stack(
        tensors=[
            torch.stack(tensors=[cos_theta, sin_theta], dim=-1),
            torch.stack(tensors=[-sin_theta, cos_theta], dim=-1),
        ],
        dim=-1,
    )  # (batch_size, 2, 2)
    rotated = torch.bmm(
        input=rotation_inv, mat2=action[:, :2].unsqueeze(dim=-1)
    ).squeeze(dim=-1)  # (batch_size, 2)
    return rotated + object_xy  # (batch_size, 2)


def distance(
    *,
    states: torch.Tensor,  # (N, 5)
    query: torch.Tensor,  # (1, 5)
) -> torch.Tensor:  # (N,)
    """Euclidean distance in ``(x, y, obj_x, obj_y)`` plus wrapped angle."""
    position_difference = states[:, :4] - query[:, :4]  # (N, 4)
    position_distance_squared = torch.sum(input=position_difference**2, dim=1)  # (N,)
    angle_difference = states[:, 4] - query[:, 4]  # (N,)
    absolute_angle = torch.abs(input=angle_difference)  # (N,)
    wrapped_angle = torch.min(input=absolute_angle, other=2.0 - absolute_angle)  # (N,)
    return torch.sqrt(input=position_distance_squared + wrapped_angle**2)  # (N,)
