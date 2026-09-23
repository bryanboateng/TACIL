import torch

# Float32 simulator observations contain round-off at roughly 1e-8 even in
# geometrically constant dimensions. Do not amplify that noise into features.
_CONSTANT_DIMENSION_THRESHOLD = 1e-12
_MINIMUM_OBSERVATION_STD = 1e-6


class MinMaxNormalizer(torch.nn.Module):
    scale: torch.Tensor  # (dimensionality,)
    offset: torch.Tensor  # (dimensionality,)

    def __init__(
        self,
        *,
        value_batch: torch.Tensor,  # (batch_size, dimensionality)
        minimum_range: float = _CONSTANT_DIMENSION_THRESHOLD,
    ) -> None:
        super().__init__()
        if minimum_range <= 0:
            raise ValueError("minimum_range must be positive")
        minimum = value_batch.min(dim=0).values  # (dimensionality,)
        value_range = value_batch.max(dim=0).values - minimum  # (dimensionality,)
        constant_dimension_mask = value_range < minimum_range  # (dimensionality,)

        scale = 2.0 / value_range  # (dimensionality,)
        scale[constant_dimension_mask] = 1.0
        offset = -1.0 - scale * minimum  # (dimensionality,)
        offset[constant_dimension_mask] = -minimum[constant_dimension_mask]

        self.register_buffer(name="scale", tensor=scale)
        self.register_buffer(name="offset", tensor=offset)

    def normalize(
        self,
        *,
        value_batch: torch.Tensor,  # (batch_size, dimensionality)
    ) -> torch.Tensor:  # (batch_size, dimensionality)
        return value_batch * self.scale + self.offset

    def denormalize(
        self,
        *,
        value_batch: torch.Tensor,  # (batch_size, dimensionality)
    ) -> torch.Tensor:  # (batch_size, dimensionality)
        return (value_batch - self.offset) / self.scale


class ZScoreNormalizer(torch.nn.Module):
    mean: torch.Tensor  # (dimensionality,)
    std: torch.Tensor  # (dimensionality,)

    def __init__(
        self,
        *,
        value_batch: torch.Tensor,  # (batch_size, dimensionality)
    ) -> None:
        super().__init__()
        std = value_batch.std(dim=0, correction=0)  # (dimensionality,)
        std[std < _MINIMUM_OBSERVATION_STD] = 1.0
        self.register_buffer(name="mean", tensor=value_batch.mean(dim=0))
        self.register_buffer(name="std", tensor=std)

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
