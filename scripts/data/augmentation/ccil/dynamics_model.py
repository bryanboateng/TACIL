import warnings
from typing import Literal

import torch

ActivationFunctionName = Literal["relu", "silu", "gelu"]

# Lipschitz constant (sup_x |f'(x)|) of each activation, used to keep the spectral-
# norm budget honest. relu is exactly 1; silu/gelu overshoot slightly above 1 and
# would otherwise inflate the realized network Lipschitz constant above the target.
# Computed numerically (coarse scan + bounded refine) to ~1e-9; regenerate if the
# activation set changes. gelu uses the torch default (exact/erf) variant.
_ACTIVATION_LIPSCHITZ: dict[ActivationFunctionName, float] = {
    "relu": 1.0,
    "silu": 1.099839320,
    "gelu": 1.128904145,
}


class DynamicsModel(torch.nn.Module):
    _state_mean: torch.Tensor  # (state_dimensionality,)
    _state_std: torch.Tensor  # (state_dimensionality,)
    _action_mean: torch.Tensor  # (action_dimensionality,)
    _action_std: torch.Tensor  # (action_dimensionality,)
    _next_state_residual_mean: torch.Tensor  # (state_dimensionality,)
    _next_state_residual_std: torch.Tensor  # (state_dimensionality,)

    def __init__(
        self,
        state_dimensionality: int,
        action_dimensionality: int,
        hidden_dimensionalities: tuple[int, ...],
        activation_function_name: ActivationFunctionName,
        lipschitz_constant_upper_bound: float | None,
    ):
        super().__init__()

        self.state_dimensionality = state_dimensionality
        self.action_dimensionality = action_dimensionality
        self.hidden_dimensionalities = hidden_dimensionalities
        self.activation_function_name = activation_function_name
        self.lipschitz_constant_upper_bound = lipschitz_constant_upper_bound

        self._network = _create_network(
            input_dimensionality=state_dimensionality + action_dimensionality,
            output_dimensionality=state_dimensionality,
            hidden_dimensionalities=hidden_dimensionalities,
            activation_function_name=activation_function_name,
            lipschitz_constant_upper_bound=lipschitz_constant_upper_bound,
        )

        self.register_buffer(
            name="_state_mean", tensor=torch.zeros(state_dimensionality)
        )
        self.register_buffer(name="_state_std", tensor=torch.ones(state_dimensionality))
        self.register_buffer(
            name="_action_mean", tensor=torch.zeros(action_dimensionality)
        )
        self.register_buffer(
            name="_action_std", tensor=torch.ones(action_dimensionality)
        )
        self.register_buffer(
            name="_next_state_residual_mean", tensor=torch.zeros(state_dimensionality)
        )
        self.register_buffer(
            name="_next_state_residual_std", tensor=torch.ones(state_dimensionality)
        )

    @torch.no_grad()
    def compute_normalization_stats(
        self,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
        next_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    ) -> None:
        self._state_mean.copy_(state_batch.mean(dim=0))
        self._state_std.copy_(self._robust_std(batch=state_batch, name="state"))

        self._action_mean.copy_(action_batch.mean(dim=0))
        self._action_std.copy_(self._robust_std(batch=action_batch, name="action"))

        residual_batch = next_state_batch - state_batch
        self._next_state_residual_mean.copy_(residual_batch.mean(dim=0))
        self._next_state_residual_std.copy_(
            self._robust_std(batch=residual_batch, name="next-state residual")
        )

    @staticmethod
    def _robust_std(
        batch: torch.Tensor,  # (batch_size, dimensionality)
        name: str,
        relative_floor: float = 1e-3,
        absolute_floor: float = 1e-6,
    ) -> torch.Tensor:  # (dimensionality,)
        # Because normalization divides by std, a near-constant dimension (std ~ 0)
        # would otherwise get its residual noise amplified without bound, letting a
        # dimension that barely moves dominate the normalized loss/metric. Floor std
        # relative to the largest std (and at an absolute minimum), which caps the
        # condition number of the normalization at 1 / relative_floor.
        std = batch.std(dim=0)
        floor = max(absolute_floor, relative_floor * std.max().item())

        degenerate_dimension_indices = torch.nonzero(std < floor, as_tuple=False)
        if degenerate_dimension_indices.numel() > 0:
            warnings.warn(
                f"{name}: {degenerate_dimension_indices.numel()} near-constant "
                f"dimension(s) {degenerate_dimension_indices.flatten().tolist()} have "
                f"std below the floor {floor:.3e} and were clamped. In normalized "
                f"space these dimensions carry little signal; consider dropping them.",
                stacklevel=2,
            )

        return std.clamp_min(floor)

    @property
    def state_std(self) -> torch.Tensor:  # (state_dimensionality,)
        return self._state_std

    @property
    def action_std(self) -> torch.Tensor:  # (action_dimensionality,)
        return self._action_std

    def normalize_state(
        self,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        return self._normalize(
            input_batch=state_batch,
            mean_batch=self._state_mean,
            std_batch=self._state_std,
        )

    def normalize_action(
        self,
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, action_dimensionality)
        return self._normalize(
            input_batch=action_batch,
            mean_batch=self._action_mean,
            std_batch=self._action_std,
        )

    def normalized_residual_from_normalized(
        self,
        normalized_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        normalized_action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        # Spectral norm bounds the Jacobian of this map, so this is
        # the function to differentiate when measuring the local Lipschitz
        # constant.
        return self._network(
            torch.cat(tensors=[normalized_state_batch, normalized_action_batch], dim=-1)
        )

    def normalized_residual(
        self,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        return self.normalized_residual_from_normalized(
            normalized_state_batch=self.normalize_state(state_batch),
            normalized_action_batch=self.normalize_action(action_batch),
        )

    def forward(
        self,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        # Returns the raw state change f̂(s, a), NOT the next state.
        return self._denormalize(
            input_batch=self.normalized_residual(
                state_batch=state_batch, action_batch=action_batch
            ),
            mean_batch=self._next_state_residual_mean,
            std_batch=self._next_state_residual_std,
        )

    def predict_next_state(
        self,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        return state_batch + self(state_batch, action_batch)

    @staticmethod
    def _normalize(
        input_batch: torch.Tensor,  # (batch_size, dimensionality)
        mean_batch: torch.Tensor,  # (dimensionality,)
        std_batch: torch.Tensor,  # (dimensionality,)
    ) -> torch.Tensor:  # (batch_size, dimensionality)
        return (input_batch - mean_batch) / std_batch

    @staticmethod
    def _denormalize(
        input_batch: torch.Tensor,  # (batch_size, dimensionality)
        mean_batch: torch.Tensor,  # (dimensionality,)
        std_batch: torch.Tensor,  # (dimensionality,)
    ) -> torch.Tensor:  # (batch_size, dimensionality)
        return input_batch * std_batch + mean_batch


def _get_activation_module_class(
    activation_function_name: ActivationFunctionName,
) -> type[torch.nn.Module]:
    match activation_function_name:
        case "relu":
            return torch.nn.ReLU
        case "silu":
            return torch.nn.SiLU
        case "gelu":
            return torch.nn.GELU


class _SpectralNormalizedLinear(torch.nn.Module):
    def __init__(
        self,
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


def _create_network(
    input_dimensionality: int,
    output_dimensionality: int,
    hidden_dimensionalities: tuple[int, ...],
    activation_function_name: ActivationFunctionName,
    lipschitz_constant_upper_bound: float | None,
) -> torch.nn.Sequential:
    activation_module_class = _get_activation_module_class(
        activation_function_name=activation_function_name
    )

    layer_dimensionalities = (
        input_dimensionality,
        *hidden_dimensionalities,
        output_dimensionality,
    )
    linear_layer_count = len(layer_dimensionalities) - 1
    activation_count = linear_layer_count - 1

    # Distribute the global Lipschitz budget across the linear layers, accounting for
    # the activations in between: with each linear layer constrained to operator norm
    # c and each activation contributing a factor kappa, the network is bounded by
    # c**linear_layer_count * kappa**activation_count, so solve that for c. With relu
    # (kappa = 1) this reduces to the plain bound ** (1 / linear_layer_count).
    if lipschitz_constant_upper_bound is None:
        lipschitz_constant_upper_bound_per_layer = None
    else:
        activation_lipschitz = _ACTIVATION_LIPSCHITZ[activation_function_name]
        lipschitz_constant_upper_bound_per_layer = (
            lipschitz_constant_upper_bound / activation_lipschitz**activation_count
        ) ** (1.0 / linear_layer_count)

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
            layers.append(activation_module_class())

    return torch.nn.Sequential(*layers)


def _create_linear_layer(
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
