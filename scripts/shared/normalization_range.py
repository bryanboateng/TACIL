import torch


class NormalizationRange:
    """Symmetric min/max normalizer mapping a feature vector to ``[-1, 1]``.

    Guards against zero-range features so constant dimensions stay finite.
    """

    def __init__(
        self,
        *,
        minimum: torch.Tensor,  # (dimensionality,)
        maximum: torch.Tensor,  # (dimensionality,)
    ) -> None:
        self.minimum = minimum  # (dimensionality,)
        self.maximum = maximum  # (dimensionality,)

    @classmethod
    def from_values(
        cls,
        *,
        values: torch.Tensor,  # (*, dimensionality) — any leading batch dimensions
    ) -> "NormalizationRange":
        flattened = values.reshape(
            -1, values.shape[-1]
        )  # (total_count, dimensionality)

        return cls(
            minimum=flattened.min(dim=0).values,  # (dimensionality,)
            maximum=flattened.max(dim=0).values,  # (dimensionality,)
        )

    def normalize(
        self,
        *,
        values: torch.Tensor,  # (*, dimensionality)
    ) -> torch.Tensor:  # (*, dimensionality) in [-1, 1]
        value_range = (self.maximum - self.minimum).clamp(min=1e-8)
        unit = (values - self.minimum) / value_range
        return unit * 2.0 - 1.0

    def denormalize(
        self,
        *,
        values: torch.Tensor,  # (*, dimensionality) in [-1, 1]
    ) -> torch.Tensor:  # (*, dimensionality)
        unit = (values + 1.0) / 2.0
        return unit * (self.maximum - self.minimum) + self.minimum
