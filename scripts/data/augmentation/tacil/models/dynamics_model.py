import torch

from scripts.data.augmentation.tacil.models.lipschitz_constrained_linear import (
    LipschitzConstrainedLinear,
)
from scripts.data.augmentation.tacil.models.normalizers import (
    Normalizers,
    agent_relative_input_dimensionality,
)

# Maximum factor by which SiLU stretches space, i.e. its (global) Lipschitz constant.
_SILU_LIPSCHITZ_CONSTANT = 1.099839320


class DynamicsModel(torch.nn.Module):
    def __init__(
        self,
        *,
        state_dimensionality: int,
        action_dimensionality: int,
        hidden_dimensionalities: tuple[int, ...],
        lipschitz_constant_upper_bound: float | None,
        agent_relative_inputs: bool = False,
    ):
        super().__init__()

        self.state_dimensionality = state_dimensionality
        self.action_dimensionality = action_dimensionality
        self.hidden_dimensionalities = hidden_dimensionalities
        self.lipschitz_constant_upper_bound = lipschitz_constant_upper_bound
        self.agent_relative_inputs = agent_relative_inputs

        self._network = _create_network(
            input_dimensionality=(
                agent_relative_input_dimensionality(
                    state_dimensionality=state_dimensionality,
                    action_dimensionality=action_dimensionality,
                )
                if agent_relative_inputs
                else state_dimensionality + action_dimensionality
            ),
            output_dimensionality=state_dimensionality,
            hidden_dimensionalities=hidden_dimensionalities,
            lipschitz_constant_upper_bound=lipschitz_constant_upper_bound,
        )

    def forward(
        self,
        normalized_input_batch: torch.Tensor,  # (batch_size, state_dimensionality + action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        return self._network(normalized_input_batch)

    def normalized_input(
        self,
        *,
        normalizers: Normalizers,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, input_dimensionality)
        if self.agent_relative_inputs:
            return normalizers.agent_relative_input(
                state_batch=state_batch, action_batch=action_batch
            )
        return normalizers.normalized_input(
            state_batch=state_batch, action_batch=action_batch
        )

    def predict_next_state(
        self,
        *,
        normalizers: Normalizers,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        normalized_state_displacement_batch = self._network(
            self.normalized_input(
                normalizers=normalizers,
                state_batch=state_batch,
                action_batch=action_batch,
            )
        )  # (batch_size, state_dimensionality)

        state_displacement_batch = normalizers.denormalize_state_displacement(
            normalized_state_displacement_batch=normalized_state_displacement_batch
        )  # (batch_size, state_dimensionality)
        return (
            state_batch + state_displacement_batch
        )  # (batch_size, state_dimensionality)

    def local_lipschitz_constants(
        self,
        *,
        normalizers: Normalizers,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size,)
        normalized_input_batch = self.normalized_input(
            normalizers=normalizers,
            state_batch=state_batch,
            action_batch=action_batch,
        )  # (batch_size, input_dimensionality)
        jacobian_batch: torch.Tensor = torch.vmap(
            func=torch.func.jacrev(func=self._network)
        )(
            normalized_input_batch
        )  # (batch_size, state_dimensionality, state_dimensionality + action_dimensionality)

        return torch.linalg.matrix_norm(jacobian_batch, ord=2)  # (batch_size,)

    def local_transition_lipschitz_constants(
        self,
        *,
        normalizers: Normalizers,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size,)
        """Spectral norm of d(normalized next state) / d(normalized state).

        Unlike local_lipschitz_constants, which measures only the steepness of
        the predicted displacement, this includes the residual identity term:
        it is the gain of the full state-to-state map, the factor by which one
        transition stretches a perturbation of the state it starts from.
        """

        def normalized_next_state(
            normalized_state: torch.Tensor,  # (state_dimensionality,)
            action: torch.Tensor,  # (action_dimensionality,)
        ) -> torch.Tensor:  # (state_dimensionality,)
            next_state_batch = self.predict_next_state(
                normalizers=normalizers,
                state_batch=normalizers.denormalize_state(
                    normalized_state_batch=normalized_state.unsqueeze(dim=0)
                ),
                action_batch=action.unsqueeze(dim=0),
            )  # (1, state_dimensionality)
            return normalizers.normalize_state(state_batch=next_state_batch).squeeze(
                dim=0
            )  # (state_dimensionality,)

        jacobian_batch = torch.vmap(
            func=torch.func.jacrev(func=normalized_next_state, argnums=0)
        )(
            normalizers.normalize_state(state_batch=state_batch), action_batch
        )  # (batch_size, state_dimensionality, state_dimensionality)

        return torch.linalg.matrix_norm(jacobian_batch, ord=2)  # (batch_size,)


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
        None
        if lipschitz_constant_upper_bound is None
        else _calculate_lipschitz_constant_upper_bound_per_layer(
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
    *, lipschitz_constant_upper_bound: float, linear_layer_count: int
) -> float | None:
    activation_count = linear_layer_count - 1

    # We want the whole network's Lipschitz constant to be at most the target bound.
    # Since Lipschitz constants multiply through a composition, the total is
    # at most c^linear_layer_count * _SILU_LIPSCHITZ_CONSTANT^activation_count,
    # where c is the norm we assign to each linear layer.
    # We solve this for c, which is what this function returns.

    return (
        lipschitz_constant_upper_bound / (_SILU_LIPSCHITZ_CONSTANT**activation_count)
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
    return LipschitzConstrainedLinear(
        input_dimensionality=input_dimensionality,
        output_dimensionality=output_dimensionality,
        lipschitz_constant_upper_bound=lipschitz_constant_upper_bound_per_layer,
    )
