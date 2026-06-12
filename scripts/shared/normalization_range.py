import torch


class NormalizationRange:
    """Symmetric min/max normalizer mapping a feature vector to ``[-1, 1]``.

    Guards against zero-range features so constant dimensions stay finite.
    """

    def __init__(self, *, minimum: torch.Tensor, maximum: torch.Tensor) -> None:
        self.minimum = minimum
        self.maximum = maximum

    @classmethod
    def from_values(cls, *, values: torch.Tensor) -> "NormalizationRange":
        flattened = values.reshape(-1, values.shape[-1])

        return cls(
            minimum=flattened.min(dim=0).values, maximum=flattened.max(dim=0).values
        )

    def normalize(self, *, values: torch.Tensor) -> torch.Tensor:
        value_range = (self.maximum - self.minimum).clamp(min=1e-8)
        unit = (values - self.minimum) / value_range
        return unit * 2.0 - 1.0

    def denormalize(self, *, values: torch.Tensor) -> torch.Tensor:
        unit = (values + 1.0) / 2.0
        return unit * (self.maximum - self.minimum) + self.minimum
