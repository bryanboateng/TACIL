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
        value_batch: torch.Tensor,  # (batch_size, input_dimensionality) or
        # (ensemble_size, batch_size, input_dimensionality)
    ) -> torch.Tensor:  # (ensemble_size, batch_size, output_dimensionality)
        if value_batch.dim() == 2:
            product = torch.einsum("ij,bjk->bik", value_batch, self.weight)
        else:
            product = torch.einsum("bij,bjk->bik", value_batch, self.weight)

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
        input_batch: torch.Tensor,  # (batch_size, input_dimensionality) or
        # (ensemble_size, batch_size, input_dimensionality)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # each (ensemble_size, batch_size, target_dimensionality)
        output_batch = input_batch
        for layer in self.hidden_layers:
            output_batch = self.activation(layer(output_batch))

        mean_batch, log_variance_batch = torch.chunk(
            input=self.output_layer(output_batch), chunks=2, dim=-1
        )

        return mean_batch, _soft_clamp(
            value_batch=log_variance_batch,
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
    """Centres and scales the model inputs, the way the reference does.

    Fitted and applied on the device, so that neither the dynamics training
    nor an imagined step has to travel to the processor for it.
    """

    def __init__(
        self,
        *,
        value_batch: torch.Tensor,  # (batch_size, dimensionality)
    ) -> None:
        self.mean = value_batch.mean(dim=0, keepdim=True)
        standard_deviation = value_batch.std(dim=0, unbiased=False, keepdim=True)
        # A dimension that never moves would otherwise divide by nothing. The
        # reference uses the same bound.
        standard_deviation[standard_deviation < 1e-12] = 1.0
        self.standard_deviation = standard_deviation

    def normalize(
        self,
        *,
        value_batch: torch.Tensor,  # (batch_size, dimensionality)
    ) -> torch.Tensor:  # (batch_size, dimensionality)
        return (value_batch - self.mean) / self.standard_deviation


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
        """Takes one step through the learned dynamics, penalty included.

        Everything stays on the device. The models predict a change in state
        and a reward, one draw is taken from a randomly chosen elite, and the
        reward is lowered by how far the ensemble disagrees.
        """
        with torch.no_grad():
            input_batch = self.input_normalizer.normalize(
                value_batch=torch.cat(tensors=[state_batch, action_batch], dim=-1)
            )

            mean_batch, log_variance_batch = self.model_ensemble(input_batch)
            # each (ensemble_size, batch_size, target_dimensionality)

            # The models predict the change, so the state is added back. The
            # last entry is the reward and stays as predicted.
            mean_batch = torch.cat(
                tensors=[mean_batch[..., :-1] + state_batch, mean_batch[..., -1:]],
                dim=-1,
            )
            # sqrt(exp(v)) written as exp(v/2), which is one operation.
            standard_deviation_batch = (0.5 * log_variance_batch).exp()

            member_sample_batch = (
                mean_batch
                + torch.randn_like(input=mean_batch) * standard_deviation_batch
            )

            batch_size = state_batch.shape[0]
            member_index_batch = self.model_ensemble.draw_elite_indices(
                count=batch_size
            )
            sample_batch = member_sample_batch[
                member_index_batch, torch.arange(end=batch_size, device=self.device)
            ]  # (batch_size, target_dimensionality)

            next_state_batch = sample_batch[..., :-1]

            penalty_batch = (
                standard_deviation_batch.norm(dim=2).amax(dim=0).unsqueeze(dim=1)
            )  # (batch_size, 1)

            return ImaginedStep(
                next_state_batch=next_state_batch,
                reward_batch=sample_batch[..., -1:]
                - self.penalty_coefficient * penalty_batch,
                termination_batch=self.termination_function(next_state_batch)
                .reshape(-1, 1)
                .float(),
                # Summed rather than averaged: the caller adds the steps up and
                # divides once, which saves waiting for the device every step.
                penalty_sum=penalty_batch.sum(),
            )


def _soft_clamp(
    *, value_batch: torch.Tensor, minimum: torch.Tensor, maximum: torch.Tensor
) -> torch.Tensor:
    clamped_batch = maximum - functional.softplus(input=maximum - value_batch)
    return minimum + functional.softplus(input=clamped_batch - minimum)
