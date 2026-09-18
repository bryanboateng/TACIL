import torch

FULL_TURN = 2.0 * torch.pi


def add(
    *,
    first_addend_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    second_addend_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    angle_mask: torch.Tensor,  # (state_dimensionality,)
) -> torch.Tensor:  # (batch_size, state_dimensionality)
    sum_batch = first_addend_batch + second_addend_batch
    return torch.where(angle_mask, sum_batch % FULL_TURN, sum_batch)


def subtract(
    *,
    minuend_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    subtrahend_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    angle_mask: torch.Tensor,  # (state_dimensionality,)
) -> torch.Tensor:  # (batch_size, state_dimensionality)
    difference_batch = minuend_batch - subtrahend_batch
    return torch.where(
        angle_mask,
        (difference_batch + torch.pi) % FULL_TURN - torch.pi,
        difference_batch,
    )


def average(
    *,
    state_batch: torch.Tensor,  # (member_count, batch_size, state_dimensionality)
    angle_mask: torch.Tensor,  # (state_dimensionality,)
) -> torch.Tensor:  # (batch_size, state_dimensionality)
    return torch.where(
        angle_mask,
        torch.atan2(
            input=torch.sin(input=state_batch).mean(dim=0),
            other=torch.cos(input=state_batch).mean(dim=0),
        )
        % FULL_TURN,
        state_batch.mean(dim=0),
    )


def create_mask(
    *, dimensionality: int, angle_indices: tuple[int, ...], like: torch.Tensor
) -> torch.Tensor:  # (dimensionality,)
    mask = torch.zeros(dimensionality, dtype=torch.bool, device=like.device)
    mask[list(angle_indices)] = True
    return mask
