import torch


class MinMaxNormalizer(torch.nn.Module):
    minimum: torch.Tensor  # (dimensionality,)
    maximum: torch.Tensor  # (dimensionality,)

    def __init__(
        self,
        *,
        value_batch: torch.Tensor,  # (batch_size, dimensionality)
    ) -> None:
        super().__init__()
        self.register_buffer(name="minimum", tensor=value_batch.min(dim=0).values)
        self.register_buffer(name="maximum", tensor=value_batch.max(dim=0).values)

    def normalize(
        self,
        *,
        value_batch: torch.Tensor,  # (batch_size, dimensionality)
    ) -> torch.Tensor:  # (batch_size, dimensionality)
        value_range = self.maximum - self.minimum  # (dimensionality,)
        unit_batch = (value_batch - self.minimum) / value_range.clamp(min=1e-8)
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
        value_batch: torch.Tensor,  # (batch_size, dimensionality)
        correction: int = 1,
        epsilon: float = 0.0,
    ) -> None:
        super().__init__()
        self.register_buffer(name="mean", tensor=value_batch.mean(dim=0))
        self.register_buffer(
            name="std",
            tensor=(value_batch.std(dim=0, correction=correction) + epsilon).clamp(
                min=1e-8
            ),
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

    def normalize_difference(
        self,
        *,
        difference_batch: torch.Tensor,  # (batch_size, dimensionality)
    ) -> torch.Tensor:  # (batch_size, dimensionality)
        return difference_batch / self.std

    def denormalize_difference(
        self,
        *,
        normalized_difference_batch: torch.Tensor,  # (batch_size, dimensionality)
    ) -> torch.Tensor:  # (batch_size, dimensionality)
        return normalized_difference_batch * self.std
