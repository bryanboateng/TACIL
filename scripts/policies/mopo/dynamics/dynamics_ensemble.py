import torch
from torch.nn import functional

from scripts.policies.mopo.config import DynamicsConfig


class EnsembleLinearLayer(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dimensionality: int,
        output_dimensionality: int,
        weight_decay: float,
        ensemble_size: int,
    ) -> None:
        super().__init__()

        self.weight = torch.nn.Parameter(
            data=torch.zeros(ensemble_size, input_dimensionality, output_dimensionality)
        )
        self.bias = torch.nn.Parameter(
            data=torch.zeros(ensemble_size, 1, output_dimensionality)
        )
        torch.nn.init.trunc_normal_(
            tensor=self.weight, std=1.0 / (2.0 * input_dimensionality**0.5)
        )

        self.saved_weight = torch.nn.Parameter(data=self.weight.detach().clone())
        self.saved_bias = torch.nn.Parameter(data=self.bias.detach().clone())

        self.weight_decay = weight_decay

    def forward(
        self,
        input_batch_per_member: torch.Tensor,
        # (ensemble_size, batch_size, input_dimensionality), or
        # (batch_size, input_dimensionality) for one batch every member shares
    ) -> torch.Tensor:  # (ensemble_size, batch_size, output_dimensionality)
        if input_batch_per_member.dim() == 2:
            product = torch.einsum("ij,bjk->bik", input_batch_per_member, self.weight)
        else:
            product = torch.einsum("bij,bjk->bik", input_batch_per_member, self.weight)

        return product + self.bias

    def save_members(self, *, member_indices: list[int]) -> None:
        self.saved_weight.data[member_indices] = self.weight.data[member_indices]
        self.saved_bias.data[member_indices] = self.bias.data[member_indices]

    def restore_members(self) -> None:
        self.weight.data.copy_(other=self.saved_weight.data)
        self.bias.data.copy_(other=self.saved_bias.data)

    def calculate_decay_loss(self) -> torch.Tensor:
        return self.weight_decay * 0.5 * self.weight.square().sum()


class DynamicsModelEnsemble(torch.nn.Module):
    elite_member_indices: torch.Tensor  # (elite_count,)

    def __init__(
        self,
        *,
        config: DynamicsConfig,
        state_dimensionality: int,
        action_dimensionality: int,
    ) -> None:
        super().__init__()

        self.ensemble_size = config.ensemble_size
        self.elite_count = config.elite_count

        # state change and reward
        target_dimensionality = state_dimensionality + 1

        layer_dimensionalities = [
            state_dimensionality + action_dimensionality,
            *config.hidden_layer_dimensionalities,
        ]
        self._hidden_layer_module_list = torch.nn.ModuleList(
            modules=[
                EnsembleLinearLayer(
                    input_dimensionality=input_dimensionality,
                    output_dimensionality=output_dimensionality,
                    weight_decay=weight_decay,
                    ensemble_size=config.ensemble_size,
                )
                for input_dimensionality, output_dimensionality, weight_decay in zip(
                    layer_dimensionalities[:-1],
                    layer_dimensionalities[1:],
                    config.weight_decays[:-1],
                    strict=True,
                )
            ]
        )
        self.output_layer = EnsembleLinearLayer(
            input_dimensionality=layer_dimensionalities[-1],
            output_dimensionality=2 * target_dimensionality,
            weight_decay=config.weight_decays[-1],
            ensemble_size=config.ensemble_size,
        )

        self.activation = torch.nn.SiLU()

        self.maximum_log_variance = torch.nn.Parameter(
            data=torch.ones(target_dimensionality) * 0.5
        )
        self.minimum_log_variance = torch.nn.Parameter(
            data=torch.ones(target_dimensionality) * -10.0
        )

        self.register_buffer(
            name="elite_member_indices", tensor=torch.arange(end=config.elite_count)
        )

    @property
    def hidden_layers(self) -> tuple[EnsembleLinearLayer, ...]:
        layers: list[EnsembleLinearLayer] = []
        for module in self._hidden_layer_module_list:
            if not isinstance(module, EnsembleLinearLayer):
                raise TypeError(
                    f"Expected an EnsembleLinearLayer member, got "
                    f"{type(module).__name__}"
                )
            layers.append(module)
        return tuple(layers)

    def forward(
        self,
        input_batch_per_member: torch.Tensor,
        # (ensemble_size, batch_size, input_dimensionality), or
        # (batch_size, input_dimensionality) for one batch every member shares
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # each (ensemble_size, batch_size, target_dimensionality)
        output_batch_per_member = input_batch_per_member
        for layer in self.hidden_layers:
            output_batch_per_member = self.activation(layer(output_batch_per_member))

        mean_batch_per_member, log_variance_batch_per_member = torch.chunk(
            input=self.output_layer(output_batch_per_member), chunks=2, dim=-1
        )

        return mean_batch_per_member, _soft_clamp(
            value=log_variance_batch_per_member,
            minimum=self.minimum_log_variance,
            maximum=self.maximum_log_variance,
        )

    def save_members(self, *, member_indices: list[int]) -> None:
        for layer in self.hidden_layers:
            layer.save_members(member_indices=member_indices)
        self.output_layer.save_members(member_indices=member_indices)

    def restore_members(self) -> None:
        for layer in self.hidden_layers:
            layer.restore_members()
        self.output_layer.restore_members()

    def calculate_decay_loss(self) -> torch.Tensor:
        decay_loss = self.output_layer.calculate_decay_loss()
        for layer in self.hidden_layers:
            decay_loss = decay_loss + layer.calculate_decay_loss()
        return decay_loss

    def set_elites(self, *, member_indices: list[int]) -> None:
        self.elite_member_indices = torch.tensor(
            data=member_indices, device=self.elite_member_indices.device
        )

    def draw_elite_indices(self, *, count: int) -> torch.Tensor:  # (count,)
        return self.elite_member_indices[
            torch.randint(
                low=0,
                high=len(self.elite_member_indices),
                size=(count,),
                device=self.elite_member_indices.device,
            )
        ]


def _soft_clamp(
    *, value: torch.Tensor, minimum: torch.Tensor, maximum: torch.Tensor
) -> torch.Tensor:
    clamped = maximum - functional.softplus(input=maximum - value)
    return minimum + functional.softplus(input=clamped - minimum)
