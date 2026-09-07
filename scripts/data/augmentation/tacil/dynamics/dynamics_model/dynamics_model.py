import torch

from scripts.data.augmentation.tacil.dynamics.dynamics_model.lipschitz_constrained_linear import (
    LipschitzConstrainedLinear,
)
from scripts.data.augmentation.tacil.dynamics.network_input import (
    NETWORK_INPUT_DIMENSIONALITY,
    create_raw_network_input,
)
from scripts.shared.normalization import ZScoreNormalizer
from scripts.shared.pusht_state import STATE_DIMENSIONALITY, add_states


class DynamicsModel(torch.nn.Module):
    def __init__(
        self,
        *,
        hidden_layer_count: int,
        hidden_layer_dimensionality: int,
        lipschitz_limit: float | None,
        input_normalizer: ZScoreNormalizer,
        state_displacement_normalizer: ZScoreNormalizer,
    ):
        super().__init__()

        self.input_normalizer = input_normalizer
        self.state_displacement_normalizer = state_displacement_normalizer

        self._network = _create_network(
            input_dimensionality=NETWORK_INPUT_DIMENSIONALITY,
            output_dimensionality=STATE_DIMENSIONALITY,
            hidden_layer_count=hidden_layer_count,
            hidden_layer_dimensionality=hidden_layer_dimensionality,
            lipschitz_limit=lipschitz_limit,
        )

    def predict_next_state(
        self, *, state_batch: torch.Tensor, action_batch: torch.Tensor
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        normalized_state_displacement_batch = self._network(
            _create_normalized_network_input(
                state_batch=state_batch,
                action_batch=action_batch,
                normalizer=self.input_normalizer,
            )
        )  # (batch_size, state_dimensionality)

        state_displacement_batch = self.state_displacement_normalizer.denormalize(
            value_batch=normalized_state_displacement_batch
        )  # (batch_size, state_dimensionality)

        return add_states(
            first_addend_batch=state_batch, second_addend_batch=state_displacement_batch
        )  # (batch_size, state_dimensionality)

    def calculate_local_lipschitz_constants(
        self, *, state_batch: torch.Tensor, action_batch: torch.Tensor
    ) -> torch.Tensor:  # (batch_size,)
        network_input_batch = _create_normalized_network_input(
            state_batch=state_batch,
            action_batch=action_batch,
            normalizer=self.input_normalizer,
        )  # (batch_size, input_dimensionality)

        jacobian_batch: torch.Tensor = torch.vmap(
            func=torch.func.jacrev(func=self._network)
        )(
            network_input_batch
        )  # (batch_size, state_dimensionality, state_dimensionality + action_dimensionality)

        return torch.linalg.matrix_norm(jacobian_batch, ord=2)  # (batch_size,)


# ---- Private --------------------------------------------


# Maximum factor by which SiLU stretches space, i.e. its Lipschitz constant.
_SILU_LIPSCHITZ_CONSTANT = 1.099839320


def _create_network(
    *,
    input_dimensionality: int,
    output_dimensionality: int,
    hidden_layer_count: int,
    hidden_layer_dimensionality: int,
    lipschitz_limit: float | None,
) -> torch.nn.Sequential:
    layer_dimensionalities = (
        input_dimensionality,
        *(hidden_layer_dimensionality,) * hidden_layer_count,
        output_dimensionality,
    )
    linear_layer_count = len(layer_dimensionalities) - 1

    lipschitz_limit_per_layer = (
        None
        if lipschitz_limit is None
        else _calculate_lipschitz_limit_per_layer(
            lipschitz_limit=lipschitz_limit, linear_layer_count=linear_layer_count
        )
    )

    layers: list[torch.nn.Module] = []
    for layer_index in range(linear_layer_count):
        layers.append(
            _create_linear_layer(
                input_dimensionality=layer_dimensionalities[layer_index],
                output_dimensionality=layer_dimensionalities[layer_index + 1],
                lipschitz_limit_per_layer=lipschitz_limit_per_layer,
            )
        )
        if layer_index != linear_layer_count - 1:
            layers.append(torch.nn.SiLU())

    return torch.nn.Sequential(*layers)


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


def _create_linear_layer(
    *,
    input_dimensionality: int,
    output_dimensionality: int,
    lipschitz_limit_per_layer: float | None,
) -> torch.nn.Module:
    if lipschitz_limit_per_layer is None:
        return torch.nn.Linear(
            in_features=input_dimensionality, out_features=output_dimensionality
        )
    else:
        return LipschitzConstrainedLinear(
            input_dimensionality=input_dimensionality,
            output_dimensionality=output_dimensionality,
            lipschitz_limit=lipschitz_limit_per_layer,
        )


def _create_normalized_network_input(
    *,
    state_batch: torch.Tensor,
    action_batch: torch.Tensor,
    normalizer: ZScoreNormalizer,
) -> torch.Tensor:  # (batch_size, NETWORK_INPUT_DIMENSIONALITY)
    return normalizer.normalize(
        value_batch=create_raw_network_input(
            state_batch=state_batch, action_batch=action_batch
        )
    )
