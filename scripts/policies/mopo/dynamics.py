"""Dynamics model ensemble of MOPO.

Ported from the reference implementation in `~/uni/masterarbeit/OfflineRL-Kit`
(`offlinerlkit/modules/dynamics_module.py`, `offlinerlkit/nets/ensemble_linear.py`
and `offlinerlkit/dynamics/ensemble_dynamics.py`).

Divergence from the reference: the termination function takes the next state
alone. The reference passes state, action and next state, because its Mujoco
termination rules read all three. The PushT rule reads the next state only.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from numpy import typing as npt
from torch.nn import functional

from scripts.policies.mopo.config import DynamicsConfig


@dataclass(frozen=True)
class ImaginedStep:
    next_state_batch: npt.NDArray[np.float32]  # (batch_size, state_dimensionality)
    reward_batch: npt.NDArray[np.float32]  # (batch_size, 1)
    termination_batch: npt.NDArray[np.float32]  # (batch_size, 1)
    penalty_mean: float


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

    def draw_elite_indices(
        self, *, count: int, generator: np.random.Generator
    ) -> npt.NDArray[np.int64]:  # (count,)
        return generator.choice(a=self.elite_indices.cpu().numpy(), size=count)


class InputNormalizer:
    def __init__(
        self,
        *,
        value_batch: npt.NDArray[np.float32],  # (batch_size, dimensionality)
    ) -> None:
        self.mean = value_batch.mean(axis=0, keepdims=True)

        std = value_batch.std(axis=0, keepdims=True)
        std[std < 1e-12] = 1.0
        self.std = std

    def normalize(
        self,
        *,
        value_batch: npt.NDArray[np.float32],  # (batch_size, dimensionality)
    ) -> npt.NDArray[np.float32]:  # (batch_size, dimensionality)
        return ((value_batch - self.mean) / self.std).astype(np.float32)


class Dynamics:
    def __init__(
        self,
        *,
        model_ensemble: DynamicsModelEnsemble,
        optimizer: torch.optim.Optimizer,
        input_normalizer: InputNormalizer,
        termination_function: Callable[
            [npt.NDArray[np.float32]], npt.NDArray[np.bool_]
        ],
        penalty_coefficient: float,
        device: torch.device,
    ) -> None:
        self.model_ensemble = model_ensemble
        self.optimizer = optimizer
        self.input_normalizer = input_normalizer
        self.termination_function = termination_function
        self.penalty_coefficient = penalty_coefficient
        self.device = device

    @torch.no_grad()
    def imagine_step(
        self,
        *,
        state_batch: npt.NDArray[np.float32],  # (batch_size, state_dimensionality)
        action_batch: npt.NDArray[np.float32],  # (batch_size, action_dimensionality)
        generator: np.random.Generator,
    ) -> ImaginedStep:
        input_batch = self.input_normalizer.normalize(
            value_batch=np.concatenate([state_batch, action_batch], axis=-1)
        )  # (batch_size, input_dimensionality)

        mean_batch, log_variance_batch = self.model_ensemble(
            torch.as_tensor(data=input_batch, device=self.device)
        )  # each (ensemble_size, batch_size, target_dimensionality)

        mean = mean_batch.cpu().numpy()
        mean[..., :-1] += state_batch
        std = np.sqrt(np.exp(log_variance_batch.cpu().numpy()))

        member_sample_batch = (
            mean + generator.normal(size=mean.shape) * std
        )  # (ensemble_size, batch_size, target_dimensionality)

        batch_size = len(state_batch)
        sample_batch = member_sample_batch[
            self.model_ensemble.draw_elite_indices(
                count=batch_size, generator=generator
            ),
            np.arange(batch_size),
        ].astype(np.float32)  # (batch_size, target_dimensionality)

        next_state_batch = sample_batch[..., :-1]
        reward_batch = sample_batch[..., -1:]

        # Aleatoric uncertainty: the widest predicted spread in the ensemble.
        penalty_batch = np.amax(np.linalg.norm(std, axis=2), axis=0)[:, None].astype(
            np.float32
        )  # (batch_size, 1)

        return ImaginedStep(
            next_state_batch=next_state_batch,
            reward_batch=reward_batch - self.penalty_coefficient * penalty_batch,
            termination_batch=self.termination_function(next_state_batch)
            .reshape(-1, 1)
            .astype(np.float32),
            penalty_mean=float(penalty_batch.mean()),
        )


def _soft_clamp(
    *, value_batch: torch.Tensor, minimum: torch.Tensor, maximum: torch.Tensor
) -> torch.Tensor:
    clamped_batch = maximum - functional.softplus(input=maximum - value_batch)
    return minimum + functional.softplus(input=clamped_batch - minimum)
