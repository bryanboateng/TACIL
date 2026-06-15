from typing import Self

import torch


class MinMaxNormalizer(torch.nn.Module):
    minimum: torch.Tensor  # (dimensionality,)
    maximum: torch.Tensor  # (dimensionality,)

    def __init__(
        self,
        *,
        minimum: torch.Tensor,  # (dimensionality,)
        maximum: torch.Tensor,  # (dimensionality,)
    ) -> None:
        super().__init__()
        self.register_buffer("minimum", minimum)
        self.register_buffer("maximum", maximum)

    @classmethod
    def from_values(
        cls,
        *,
        value_batch: torch.Tensor,  # (batch_size, dimensionality)
    ) -> Self:
        flattened = value_batch.reshape(
            -1, value_batch.shape[-1]
        )  # (total_count, dimensionality)

        return cls(
            minimum=flattened.min(dim=0).values,  # (dimensionality,)
            maximum=flattened.max(dim=0).values,  # (dimensionality,)
        )

    def normalize(
        self,
        *,
        value_batch: torch.Tensor,  # (batch_size, dimensionality)
    ) -> torch.Tensor:  # (batch_size, dimensionality)
        value_range = (self.maximum - self.minimum).clamp(min=1e-8)  # (dimensionality,)
        unit_batch = (value_batch - self.minimum) / value_range
        return unit_batch * 2.0 - 1.0

    def denormalize(
        self,
        *,
        value_batch: torch.Tensor,  # (batch_size, dimensionality)
    ) -> torch.Tensor:  # (batch_size, dimensionality)
        unit_batch = (value_batch + 1.0) / 2.0
        return unit_batch * (self.maximum - self.minimum) + self.minimum


class ZScoreNormalizer(torch.nn.Module):
    mean: torch.Tensor  # (dimensionality,)
    std: torch.Tensor  # (dimensionality,)

    def __init__(
        self,
        *,
        mean: torch.Tensor,  # (dimensionality,)
        std: torch.Tensor,  # (dimensionality,)
    ) -> None:
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    @classmethod
    def from_values(
        cls,
        *,
        value_batch: torch.Tensor,  # (batch_size, dimensionality)
    ) -> Self:
        return cls(
            mean=value_batch.mean(dim=0),  # (dimensionality,)
            std=value_batch.std(dim=0),  # (dimensionality,)
        )

    def normalize(
        self,
        *,
        value_batch: torch.Tensor,  # (batch_size, dimensionality)
    ) -> torch.Tensor:  # (batch_size, dimensionality)
        return (value_batch - self.mean) / self.std

    def denormalize(
        self,
        *,
        value_batch: torch.Tensor,  # (batch_size, dimensionality)
    ) -> torch.Tensor:  # (batch_size, dimensionality)
        return value_batch * self.std + self.mean
