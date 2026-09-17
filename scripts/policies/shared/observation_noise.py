import torch


def disturb_half_of_the_batch(
    *,
    value_batch: torch.Tensor,  # (batch_size, dimensionality)
    noise: float,
) -> torch.Tensor:  # (batch_size, dimensionality)
    selection_mask = (
        torch.rand(value_batch.shape[0], device=value_batch.device) > 0.5
    )  # (batch_size,)

    disturbance_batch = noise * (
        2.0 * torch.rand(value_batch.shape, device=value_batch.device) - 1.0
    )  # (batch_size, dimensionality)

    return torch.where(
        selection_mask.unsqueeze(dim=1), value_batch + disturbance_batch, value_batch
    )
