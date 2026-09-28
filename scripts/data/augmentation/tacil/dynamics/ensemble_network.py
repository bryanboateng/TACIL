import math

import torch
from torch.nn import functional
from torch.nn.utils import parametrize


class EnsembleNetwork(torch.nn.Module):
    def __init__(
        self,
        *,
        member_count: int,
        layer_dimensionalities: tuple[int, ...],
        lipschitz_limit_per_layer: float | None,
    ):
        super().__init__()
        self._layers = torch.nn.ModuleList(
            modules=[
                _EnsembleLinear(
                    member_count=member_count,
                    input_dimensionality=input_dimensionality,
                    output_dimensionality=output_dimensionality,
                    lipschitz_limit=lipschitz_limit_per_layer,
                )
                for input_dimensionality, output_dimensionality in zip(
                    layer_dimensionalities[:-1], layer_dimensionalities[1:], strict=True
                )
            ]
        )

    def forward(
        self,
        input_batch: torch.Tensor,
        # (member_count, batch_size, input_dimensionality), or
        # (batch_size, input_dimensionality) for one batch every member shares
    ) -> torch.Tensor:  # (member_count, batch_size, output_dimensionality)
        weights, biases = self._collect_parameters()
        return torch.vmap(
            func=_apply_layers, in_dims=(0, 0, 0 if input_batch.dim() == 3 else None)
        )(weights, biases, input_batch)

    def calculate_jacobians(
        self,
        *,
        input_batch: torch.Tensor,  # (member_count, batch_size, input_dimensionality)
    ) -> (
        torch.Tensor
    ):  # (member_count, batch_size, output_dimensionality, input_dimensionality)
        weights, biases = self._collect_parameters()
        return torch.vmap(
            func=torch.vmap(
                func=torch.func.jacrev(func=_apply_layers, argnums=2),
                in_dims=(None, None, 0),
            ),
            in_dims=(0, 0, 0),
        )(weights, biases, input_batch)

    # ---- Private --------------------------------------------

    def _collect_parameters(
        self,
    ) -> tuple[
        list[torch.Tensor],  # weights, each (member_count, output_dim., input_dim.)
        list[torch.Tensor],  # biases, each (member_count, output_dimensionality)
    ]:
        weights: list[torch.Tensor] = []
        biases: list[torch.Tensor] = []
        for layer in self._layers:
            assert isinstance(layer, _EnsembleLinear)
            weights.append(layer.weight)
            biases.append(layer.bias)
        return weights, biases


def create_ensemble_network(
    *,
    member_count: int,
    input_dimensionality: int,
    output_dimensionality: int,
    hidden_layer_count: int,
    hidden_layer_dimensionality: int,
    lipschitz_limit: float | None,
) -> EnsembleNetwork:
    layer_dimensionalities = (
        input_dimensionality,
        *(hidden_layer_dimensionality,) * hidden_layer_count,
        output_dimensionality,
    )
    linear_layer_count = len(layer_dimensionalities) - 1

    return EnsembleNetwork(
        member_count=member_count,
        layer_dimensionalities=layer_dimensionalities,
        lipschitz_limit_per_layer=(
            None
            if lipschitz_limit is None
            else _calculate_lipschitz_limit_per_layer(
                lipschitz_limit=lipschitz_limit, linear_layer_count=linear_layer_count
            )
        ),
    )


def calculate_spectral_norms(
    *,
    matrix_batch: torch.Tensor,  # (*batch_shape, row_count, column_count)
) -> torch.Tensor:  # (*batch_shape,)
    finite_mask = (
        torch.isfinite(matrix_batch).flatten(start_dim=-2).all(dim=-1)
    )  # (*batch_shape,)
    finite_matrix_batch = torch.where(
        condition=finite_mask[..., None, None], input=matrix_batch, other=0.0
    )  # (*batch_shape, row_count, column_count)

    try:
        spectral_norms = torch.linalg.matrix_norm(finite_matrix_batch, ord=2)
    except RuntimeError:
        spectral_norms = torch.linalg.matrix_norm(finite_matrix_batch.cpu(), ord=2).to(
            device=matrix_batch.device
        )

    return torch.where(condition=finite_mask, input=spectral_norms, other=torch.nan)


# ---- Private --------------------------------------------


# Maximum factor by which SiLU stretches space, i.e. its Lipschitz constant.
_SILU_LIPSCHITZ_CONSTANT = 1.099839320


class _EnsembleLinear(torch.nn.Module):
    weight: torch.Tensor  # (member_count, output_dimensionality, input_dimensionality)
    bias: torch.Tensor  # (member_count, output_dimensionality)

    def __init__(
        self,
        *,
        member_count: int,
        input_dimensionality: int,
        output_dimensionality: int,
        lipschitz_limit: float | None,
    ):
        super().__init__()
        bound = 1.0 / math.sqrt(input_dimensionality)
        self.weight = torch.nn.Parameter(
            data=torch.empty(
                member_count, output_dimensionality, input_dimensionality
            ).uniform_(-bound, bound)
        )
        self.bias = torch.nn.Parameter(
            data=torch.empty(member_count, output_dimensionality).uniform_(
                -bound, bound
            )
        )
        if lipschitz_limit is not None:
            parametrize.register_parametrization(
                module=self,
                tensor_name="weight",
                parametrization=_SpectralNormLimit(lipschitz_limit=lipschitz_limit),
            )


class _SpectralNormLimit(torch.nn.Module):
    def __init__(self, *, lipschitz_limit: float):
        super().__init__()
        self._lipschitz_limit = lipschitz_limit

    def forward(
        self,
        weight: torch.Tensor,  # (member_count, output_dimensionality, input_dimensionality)
    ) -> torch.Tensor:  # (member_count, output_dimensionality, input_dimensionality)
        spectral_norms = calculate_spectral_norms(
            matrix_batch=weight
        )  # (member_count,)

        return (
            weight
            / torch.clamp(input=spectral_norms / self._lipschitz_limit, min=1.0)[
                :, None, None
            ]
        )


def _apply_layers(
    weights: list[torch.Tensor],  # each (output_dimensionality, input_dimensionality)
    biases: list[torch.Tensor],  # each (output_dimensionality,)
    input_batch: torch.Tensor,  # (*batch_shape, input_dimensionality)
) -> torch.Tensor:  # (*batch_shape, output_dimensionality)
    output_batch = input_batch
    for layer_index, (weight, bias) in enumerate(zip(weights, biases, strict=True)):
        output_batch = functional.linear(input=output_batch, weight=weight, bias=bias)
        if layer_index != len(weights) - 1:
            output_batch = functional.silu(input=output_batch)
    return output_batch


def _calculate_lipschitz_limit_per_layer(
    *, lipschitz_limit: float, linear_layer_count: int
) -> float:
    activation_count = linear_layer_count - 1

    # We want the whole network's Lipschitz constant to be at most the target limit.
    # Since Lipschitz constants multiply through a composition, the total is
    # at most c^linear_layer_count * _SILU_LIPSCHITZ_CONSTANT^activation_count,
    # where c is the norm we assign to each linear layer.
    # We solve this for c, which is what this function returns.

    return (lipschitz_limit / (_SILU_LIPSCHITZ_CONSTANT**activation_count)) ** (
        1.0 / linear_layer_count
    )
