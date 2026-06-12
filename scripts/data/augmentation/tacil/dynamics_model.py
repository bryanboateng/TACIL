import torch

# Maximum factor by which SiLU stretches space, i.e. its (global) Lipschitz constant.
_SILU_LIPSCHITZ_CONSTANT = 1.099839320


class DynamicsModel(torch.nn.Module):
    _state_mean: torch.Tensor  # (state_dimensionality,)
    _state_std: torch.Tensor  # (state_dimensionality,)
    _action_mean: torch.Tensor  # (action_dimensionality,)
    _action_std: torch.Tensor  # (action_dimensionality,)
    _next_state_residual_mean: torch.Tensor  # (state_dimensionality,)
    _next_state_residual_std: torch.Tensor  # (state_dimensionality,)

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

        self.register_buffer(
            name="_state_mean", tensor=torch.zeros(state_dimensionality)
        )
        self.register_buffer(
            name="_action_mean", tensor=torch.zeros(action_dimensionality)
        )
        self.register_buffer(
            name="_next_state_residual_mean", tensor=torch.zeros(state_dimensionality)
        )

        self.register_buffer(name="_state_std", tensor=torch.ones(state_dimensionality))
        self.register_buffer(
            name="_action_std", tensor=torch.ones(action_dimensionality)
        )
        self.register_buffer(
            name="_next_state_residual_std", tensor=torch.ones(state_dimensionality)
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

        self._state_mean.copy_(state_batch.mean(dim=0))
        self._action_mean.copy_(action_batch.mean(dim=0))
        self._next_state_residual_mean.copy_(next_state_residual_batch.mean(dim=0))

        self._state_std.copy_(state_batch.std(dim=0))
        self._action_std.copy_(action_batch.std(dim=0))
        self._next_state_residual_std.copy_(next_state_residual_batch.std(dim=0))

    def normalize_state(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        return self._normalize(
            input_batch=state_batch,
            mean_batch=self._state_mean,
            std_batch=self._state_std,
        )

    def denormalize_state(
        self,
        *,
        normalized_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        return self._denormalize(
            input_batch=normalized_state_batch,
            mean_batch=self._state_mean,
            std_batch=self._state_std,
        )

    def predict_next_state(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:
        normalized_residual: torch.Tensor = self._network(
            torch.cat(
                tensors=[
                    self.normalize_state(state_batch=state_batch),
                    self._normalize_action(action_batch=action_batch),
                ],
                dim=-1,
            )
        )

        raw_residual = self._denormalize(
            input_batch=normalized_residual,
            mean_batch=self._next_state_residual_mean,
            std_batch=self._next_state_residual_std,
        )
        return state_batch + raw_residual

    def local_lipschitz_constants(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:
        normalized_input_batch = torch.cat(
            tensors=[
                self.normalize_state(state_batch=state_batch),
                self._normalize_action(action_batch=action_batch),
            ],
            dim=-1,
        )
        jacobian_batch: torch.Tensor = torch.vmap(
            func=torch.func.jacrev(func=self._network)
        )(normalized_input_batch)

        return torch.linalg.matrix_norm(jacobian_batch, ord=2)

    def _normalize_action(
        self,
        *,
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, action_dimensionality)
        return self._normalize(
            input_batch=action_batch,
            mean_batch=self._action_mean,
            std_batch=self._action_std,
        )

    @staticmethod
    def _normalize(
        *,
        input_batch: torch.Tensor,  # (batch_size, dimensionality)
        mean_batch: torch.Tensor,  # (dimensionality,)
        std_batch: torch.Tensor,  # (dimensionality,)
    ) -> torch.Tensor:  # (batch_size, dimensionality)
        return (input_batch - mean_batch) / std_batch

    @staticmethod
    def _denormalize(
        *,
        input_batch: torch.Tensor,  # (batch_size, dimensionality)
        mean_batch: torch.Tensor,  # (dimensionality,)
        std_batch: torch.Tensor,  # (dimensionality,)
    ) -> torch.Tensor:  # (batch_size, dimensionality)
        return input_batch * std_batch + mean_batch


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
