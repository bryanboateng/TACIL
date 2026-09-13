from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.nn import functional

from scripts.policies.mopo.config import DynamicsConfig

# Decides, from a state alone, whether the episode is over. The dynamics
# model predicts where the robot lands but not whether it fell over. It takes
# and returns device tensors, so that a rollout never leaves the device.
type TerminationFunction = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class ImaginedStep:
    next_state_batch: torch.Tensor  # (batch_size, state_dimensionality)
    reward_batch: torch.Tensor  # (batch_size, 1)
    termination_batch: torch.Tensor  # (batch_size, 1)
    penalty_sum: torch.Tensor  # ()


class EnsembleLinearLayer(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dimensionality: int,
        output_dimensionality: int,
        ensemble_size: int,
        weight_decay: float,
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
        input_batches: torch.Tensor,
        # (ensemble_size, batch_size, input_dimensionality), or
        # (batch_size, input_dimensionality) for one batch every member shares
    ) -> torch.Tensor:  # (ensemble_size, batch_size, output_dimensionality)
        if input_batches.dim() == 2:
            product = torch.einsum("ij,bjk->bik", input_batches, self.weight)
        else:
            product = torch.einsum("bij,bjk->bik", input_batches, self.weight)

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
    elite_indices: torch.Tensor  # (elite_count,)

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

        # The model predicts the state change and the reward, each as a mean
        # and a log variance.
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
                    ensemble_size=config.ensemble_size,
                    weight_decay=weight_decay,
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
            ensemble_size=config.ensemble_size,
            weight_decay=config.weight_decays[-1],
        )

        self.activation = torch.nn.SiLU()

        self.maximum_log_variance = torch.nn.Parameter(
            data=torch.ones(target_dimensionality) * 0.5
        )
        self.minimum_log_variance = torch.nn.Parameter(
            data=torch.ones(target_dimensionality) * -10.0
        )

        self.register_buffer(
            name="elite_indices", tensor=torch.arange(end=config.elite_count)
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
        input_batches: torch.Tensor,
        # (ensemble_size, batch_size, input_dimensionality), or
        # (batch_size, input_dimensionality) for one batch every member shares
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # each (ensemble_size, batch_size, target_dimensionality)
        output_batches = input_batches
        for layer in self.hidden_layers:
            output_batches = self.activation(layer(output_batches))

        mean_batches, log_variance_batches = torch.chunk(
            input=self.output_layer(output_batches), chunks=2, dim=-1
        )

        return mean_batches, _soft_clamp(
            value=log_variance_batches,
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
        self.elite_indices = torch.tensor(
            data=member_indices, device=self.elite_indices.device
        )

    def draw_elite_indices(self, *, count: int) -> torch.Tensor:  # (count,)
        return self.elite_indices[
            torch.randint(
                low=0,
                high=len(self.elite_indices),
                size=(count,),
                device=self.elite_indices.device,
            )
        ]


class InputNormalizer:
    def __init__(
        self,
        *,
        value_batch: torch.Tensor,  # (batch_size, dimensionality)
    ) -> None:
        self.mean = value_batch.mean(dim=0, keepdim=True)
        standard_deviation = value_batch.std(dim=0, unbiased=False, keepdim=True)
        standard_deviation[standard_deviation < 1e-12] = 1.0
        self.standard_deviation = standard_deviation

    def normalize(
        self,
        *,
        value_batch: torch.Tensor,  # (batch_size, dimensionality)
    ) -> torch.Tensor:
        return (
            value_batch - self.mean
        ) / self.standard_deviation  # (batch_size, dimensionality)


class Dynamics:
    def __init__(
        self,
        *,
        model_ensemble: DynamicsModelEnsemble,
        optimizer: torch.optim.Optimizer,
        input_normalizer: InputNormalizer,
        termination_function: TerminationFunction,
        penalty_coefficient: float,
        device: torch.device,
    ) -> None:
        self.model_ensemble = model_ensemble
        self.optimizer = optimizer
        self.input_normalizer = input_normalizer
        self.termination_function = termination_function
        self.penalty_coefficient = penalty_coefficient
        self.device = device

    def imagine_step(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> ImaginedStep:
        with torch.no_grad():
            input_batch = self.input_normalizer.normalize(
                value_batch=torch.cat(tensors=[state_batch, action_batch], dim=-1)
            )

            (
                mean_batches,  # (ensemble_size, batch_size, state_dimensionality + 1)
                log_variance_batches,  # (ensemble_size, batch_size, state_dimensionality + 1)
            ) = self.model_ensemble(input_batch)
            standard_deviation_batches = (0.5 * log_variance_batches).exp()

            sample_batches = _draw_member_samples(
                state_batch=state_batch,
                mean_batches=mean_batches,
                standard_deviation_batches=standard_deviation_batches,
            )  # (ensemble_size, batch_size, state_dimensionality + 1)

            batch_size = state_batch.shape[0]
            member_index_batch = self.model_ensemble.draw_elite_indices(
                count=batch_size
            )
            sample_batch = sample_batches[
                member_index_batch, torch.arange(end=batch_size, device=self.device)
            ]  # (batch_size, target_dimensionality)

            next_state_batch = sample_batch[..., :-1]
            reward_batch = sample_batch[..., -1:]

            penalty_batch = (
                standard_deviation_batches.norm(dim=2).amax(dim=0).unsqueeze(dim=1)
            )  # (batch_size, 1)

            return ImaginedStep(
                next_state_batch=next_state_batch,
                reward_batch=reward_batch - self.penalty_coefficient * penalty_batch,
                termination_batch=self.termination_function(next_state_batch)
                .reshape(-1, 1)
                .float(),
                penalty_sum=penalty_batch.sum(),
            )


def _draw_member_samples(
    *,
    state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    mean_batches: torch.Tensor,  # (ensemble_size, batch_size, state_dimensionality + 1)
    standard_deviation_batches: torch.Tensor,  # (ensemble_size, batch_size, state_dimensionality + 1)
) -> torch.Tensor:
    next_state_mean_batches = (
        mean_batches[..., :-1] + state_batch
    )  # (ensemble_size, batch_size, state_dimensionality)
    reward_mean_batches = mean_batches[..., -1:]  # (ensemble_size, batch_size, 1)

    mean_batches = torch.cat(
        tensors=[next_state_mean_batches, reward_mean_batches], dim=-1
    )  # (ensemble_size, batch_size, target_dimensionality)
    return (
        mean_batches + torch.randn_like(input=mean_batches) * standard_deviation_batches
    )  # (ensemble_size, batch_size, state_dimensionality + 1)


def _soft_clamp(
    *, value: torch.Tensor, minimum: torch.Tensor, maximum: torch.Tensor
) -> torch.Tensor:
    clamped = maximum - functional.softplus(input=maximum - value)
    return minimum + functional.softplus(input=clamped - minimum)
