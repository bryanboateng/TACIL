import torch


class LipschitzConstrainedLinear(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dimensionality: int,
        output_dimensionality: int,
        lipschitz_limit: float,
    ):
        super().__init__()
        self._linear = torch.nn.Linear(
            in_features=input_dimensionality, out_features=output_dimensionality
        )
        torch.nn.utils.parametrize.register_parametrization(
            module=self._linear,
            tensor_name="weight",
            parametrization=_SpectralNormLimit(lipschitz_limit=lipschitz_limit),
        )

    def forward(
        self,
        input_batch: torch.Tensor,  # (batch_size, input_dimensionality)
    ) -> torch.Tensor:  # (batch_size, output_dimensionality)
        return self._linear(input_batch)


# ---- Private --------------------------------------------


class _SpectralNormLimit(torch.nn.Module):
    def __init__(self, *, lipschitz_limit: float):
        super().__init__()
        self._lipschitz_limit = lipschitz_limit

    def forward(
        self,
        weight: torch.Tensor,  # (output_dimensionality, input_dimensionality)
    ) -> torch.Tensor:  # (output_dimensionality, input_dimensionality)
        spectral_norm = torch.linalg.matrix_norm(weight, ord=2)

        return weight / torch.clamp(
            input=spectral_norm / self._lipschitz_limit, min=1.0
        )
