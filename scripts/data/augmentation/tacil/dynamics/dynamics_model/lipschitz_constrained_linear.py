import torch


class LipschitzConstrainedLinear(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dimensionality: int,
        output_dimensionality: int,
        lipschitz_constant_upper_bound: float,
    ):
        super().__init__()
        self._linear = torch.nn.Linear(
            in_features=input_dimensionality, out_features=output_dimensionality
        )
        torch.nn.utils.parametrize.register_parametrization(
            module=self._linear,
            tensor_name="weight",
            parametrization=_SpectralNormBound(
                weight=self._linear.weight,
                lipschitz_constant_upper_bound=lipschitz_constant_upper_bound,
            ),
        )

    def forward(
        self,
        input_batch: torch.Tensor,  # (batch_size, input_dimensionality)
    ) -> torch.Tensor:  # (batch_size, output_dimensionality)
        return self._linear(input_batch)


# ---- Private --------------------------------------------


class _SpectralNormBound(torch.nn.Module):
    left_singular_vector: torch.Tensor  # (output_dimensionality,)
    right_singular_vector: torch.Tensor  # (input_dimensionality,)

    def __init__(self, *, weight: torch.Tensor, lipschitz_constant_upper_bound: float):
        super().__init__()
        output_dimensionality, input_dimensionality = weight.shape
        self._lipschitz_constant_upper_bound = lipschitz_constant_upper_bound
        self.register_buffer(
            name="left_singular_vector",
            tensor=torch.nn.functional.normalize(
                input=torch.randn(output_dimensionality), dim=0
            ),
        )
        self.register_buffer(
            name="right_singular_vector",
            tensor=torch.nn.functional.normalize(
                input=torch.randn(input_dimensionality), dim=0
            ),
        )

    def forward(
        self,
        weight: torch.Tensor,  # (output_dimensionality, input_dimensionality)
    ) -> torch.Tensor:  # (output_dimensionality, input_dimensionality)
        left_singular_vector = self.left_singular_vector
        right_singular_vector = self.right_singular_vector

        if self.training:
            with torch.no_grad():
                torch.nn.functional.normalize(
                    input=weight.t() @ left_singular_vector,
                    dim=0,
                    out=self.right_singular_vector,
                )
                torch.nn.functional.normalize(
                    input=weight @ self.right_singular_vector,
                    dim=0,
                    out=self.left_singular_vector,
                )
            left_singular_vector = self.left_singular_vector.clone()
            right_singular_vector = self.right_singular_vector.clone()

        spectral_norm = torch.dot(
            input=left_singular_vector, tensor=weight @ right_singular_vector
        )
        return weight / torch.clamp(
            input=spectral_norm / self._lipschitz_constant_upper_bound, min=1.0
        )
