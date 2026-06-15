import torch

from scripts.shared.normalization_range import ZScoreNormalizer

# Maximum factor by which SiLU stretches space, i.e. its (global) Lipschitz constant.
_SILU_LIPSCHITZ_CONSTANT = 1.099839320


class DynamicsModel(torch.nn.Module):
    state_normalizer: ZScoreNormalizer
    action_normalizer: ZScoreNormalizer
    next_state_residual_normalizer: ZScoreNormalizer

    def __init__(
        self,
        *,
        state_dimensionality: int,
        action_dimensionality: int,
        hidden_dimensionalities: tuple[int, ...],
        lipschitz_constant_upper_bound: float | None,
    ):
        super().__init__()

        self.state_dimensionality = state_dimensionality
        self.action_dimensionality = action_dimensionality
        self.hidden_dimensionalities = hidden_dimensionalities
        self.lipschitz_constant_upper_bound = lipschitz_constant_upper_bound

        self._network = _create_network(
            input_dimensionality=state_dimensionality + action_dimensionality,
            output_dimensionality=state_dimensionality,
            hidden_dimensionalities=hidden_dimensionalities,
            lipschitz_constant_upper_bound=lipschitz_constant_upper_bound,
        )

        # Placeholders; overwritten by compute_normalization_stats() or load_state_dict().
        self.state_normalizer = ZScoreNormalizer(
            mean=torch.zeros(state_dimensionality), std=torch.ones(state_dimensionality)
        )
        self.action_normalizer = ZScoreNormalizer(
            mean=torch.zeros(action_dimensionality),
            std=torch.ones(action_dimensionality),
        )
        self.next_state_residual_normalizer = ZScoreNormalizer(
            mean=torch.zeros(state_dimensionality), std=torch.ones(state_dimensionality)
        )

    @torch.no_grad()
    def compute_normalization_stats(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
        next_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    ) -> None:
        next_state_residual_batch = next_state_batch - state_batch
        self.state_normalizer = ZScoreNormalizer.from_values(value_batch=state_batch)
        self.action_normalizer = ZScoreNormalizer.from_values(value_batch=action_batch)
        self.next_state_residual_normalizer = ZScoreNormalizer.from_values(
            value_batch=next_state_residual_batch
        )

    def normalize_state(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        return self.state_normalizer.normalize(value_batch=state_batch)

    def denormalize_state(
        self,
        *,
        normalized_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        return self.state_normalizer.denormalize(value_batch=normalized_state_batch)

    def predict_next_state(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        normalized_residual_batch: torch.Tensor = self._network(
            torch.cat(
                tensors=[
                    self.normalize_state(state_batch=state_batch),
                    self._normalize_action(action_batch=action_batch),
                ],
                dim=-1,
            )
        )  # (batch_size, state_dimensionality)

        raw_residual = self.next_state_residual_normalizer.denormalize(
            value_batch=normalized_residual_batch
        )  # (batch_size, state_dimensionality)
        return state_batch + raw_residual  # (batch_size, state_dimensionality)

    def local_lipschitz_constants(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size,)
        normalized_input_batch = torch.cat(
            tensors=[
                self.normalize_state(state_batch=state_batch),
                self._normalize_action(action_batch=action_batch),
            ],
            dim=-1,
        )  # (batch_size, state_dimensionality + action_dimensionality)
        jacobian_batch: torch.Tensor = torch.vmap(
            func=torch.func.jacrev(func=self._network)
        )(
            normalized_input_batch
        )  # (batch_size, state_dimensionality, state_dimensionality + action_dimensionality)

        return torch.linalg.matrix_norm(jacobian_batch, ord=2)  # (batch_size,)

    def _normalize_action(
        self,
        *,
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, action_dimensionality)
        return self.action_normalizer.normalize(value_batch=action_batch)


def _create_network(
    *,
    input_dimensionality: int,
    output_dimensionality: int,
    hidden_dimensionalities: tuple[int, ...],
    lipschitz_constant_upper_bound: float | None,
) -> torch.nn.Sequential:

    layer_dimensionalities = (
        input_dimensionality,
        *hidden_dimensionalities,
        output_dimensionality,
    )
    linear_layer_count = len(layer_dimensionalities) - 1

    lipschitz_constant_upper_bound_per_layer = (
        _calculate_lipschitz_constant_upper_bound_per_layer(
            lipschitz_constant_upper_bound=lipschitz_constant_upper_bound,
            linear_layer_count=linear_layer_count,
        )
    )

    layers: list[torch.nn.Module] = []
    for layer_index in range(linear_layer_count):
        layers.append(
            _create_linear_layer(
                input_dimensionality=layer_dimensionalities[layer_index],
                output_dimensionality=layer_dimensionalities[layer_index + 1],
                lipschitz_constant_upper_bound_per_layer=lipschitz_constant_upper_bound_per_layer,
            )
        )
        if layer_index != linear_layer_count - 1:
            layers.append(torch.nn.SiLU())

    return torch.nn.Sequential(*layers)


def _calculate_lipschitz_constant_upper_bound_per_layer(
    *, lipschitz_constant_upper_bound: float | None, linear_layer_count: int
) -> float | None:
    activation_count = linear_layer_count - 1

    if lipschitz_constant_upper_bound is None:
        return None
    else:
        # We want the whole network's Lipschitz constant to be at most the target bound.
        # Since Lipschitz constants multiply through a composition, the total is
        # at most c^linear_layer_count * _SILU_LIPSCHITZ_CONSTANT^activation_count,
        # where c is the norm we assign to each linear layer.
        # We solve this for c, which is what this function returns.

        return (
            lipschitz_constant_upper_bound
            / (_SILU_LIPSCHITZ_CONSTANT**activation_count)
        ) ** (1.0 / linear_layer_count)


def _create_linear_layer(
    *,
    input_dimensionality: int,
    output_dimensionality: int,
    lipschitz_constant_upper_bound_per_layer: float | None,
) -> torch.nn.Module:
    if lipschitz_constant_upper_bound_per_layer is None:
        return torch.nn.Linear(
            in_features=input_dimensionality, out_features=output_dimensionality
        )
    return _SpectralNormalizedLinear(
        input_dimensionality=input_dimensionality,
        output_dimensionality=output_dimensionality,
        lipschitz_constant_upper_bound=lipschitz_constant_upper_bound_per_layer,
    )


class _SpectralNormalizedLinear(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dimensionality: int,
        output_dimensionality: int,
        lipschitz_constant_upper_bound: float,
    ):
        super().__init__()
        self._linear = torch.nn.utils.parametrizations.spectral_norm(
            torch.nn.Linear(
                in_features=input_dimensionality, out_features=output_dimensionality
            )
        )
        self._lipschitz_constant_upper_bound = lipschitz_constant_upper_bound

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        return self._linear(input_batch) * self._lipschitz_constant_upper_bound
