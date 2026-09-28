from dataclasses import dataclass

import torch
from tqdm import tqdm

from scripts.data.augmentation.ccil.config import DynamicsConfig, LipschitzType


@dataclass(frozen=True)
class EpochLosses:
    total: float
    data_term: float
    regularization_term: float


class WorldModel(torch.nn.Module):
    def __init__(
        self,
        *,
        state_dimensionality: int,
        action_dimensionality: int,
        hidden_layer_sizes: tuple[int, ...],
        lipschitz_type: LipschitzType,
        lipschitz_constraint: float,
    ) -> None:
        super().__init__()
        layer_sizes = (
            state_dimensionality + action_dimensionality,
            *hidden_layer_sizes,
            state_dimensionality,
        )
        self._layers = torch.nn.ModuleList(
            modules=[
                torch.nn.Linear(
                    in_features=layer_sizes[index], out_features=layer_sizes[index + 1]
                )
                for index in range(len(layer_sizes) - 1)
            ]
        )
        self.slack_network: _SlackNetwork | None = None

        match lipschitz_type:
            case "spectral_normalization":
                for index, layer in enumerate(iterable=self._layers):
                    torch.nn.utils.parametrize.register_parametrization(
                        layer,
                        "weight",
                        _SpectralNormClamp(
                            input_dimensionality=layer_sizes[index],
                            output_dimensionality=layer_sizes[index + 1],
                            lipschitz_constraint=lipschitz_constraint,
                        ),
                    )
            case "slack":
                self.slack_network = _SlackNetwork(
                    state_dimensionality=state_dimensionality,
                    action_dimensionality=action_dimensionality,
                    hidden_layer_sizes=hidden_layer_sizes,
                )
            case "soft_sampling" | "none":
                pass

        # Transformation buffers, overwritten by initialize_transformations.
        self.register_buffer("state_shift", torch.zeros(state_dimensionality))
        self.register_buffer("state_scale", torch.ones(state_dimensionality))
        self.register_buffer("action_shift", torch.zeros(action_dimensionality))
        self.register_buffer("action_scale", torch.ones(action_dimensionality))
        self.register_buffer("residual_shift", torch.zeros(state_dimensionality))
        self.register_buffer("residual_scale", torch.ones(state_dimensionality))
        self.register_buffer("residual_mask", torch.ones(state_dimensionality))

    def initialize_transformations(
        self,
        *,
        state_batch: torch.Tensor,  # (transition_count, state_dimensionality)
        action_batch: torch.Tensor,  # (transition_count, action_dimensionality)
        next_state_batch: torch.Tensor,  # (transition_count, state_dimensionality)
    ) -> None:
        residual_batch = next_state_batch - state_batch

        self.state_shift = state_batch.mean(dim=0)
        self.state_scale = (state_batch - self.state_shift).abs().mean(dim=0)
        self.action_shift = action_batch.mean(dim=0)
        self.action_scale = (action_batch - self.action_shift).abs().mean(dim=0)
        self.residual_shift = residual_batch.mean(dim=0)
        self.residual_scale = (residual_batch - self.residual_shift).abs().mean(dim=0)
        self.residual_mask = (self.residual_scale >= 1e-8).float()

    def forward(
        self,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality), normalized residual
        features = torch.cat(
            tensors=[
                (state_batch - self.state_shift) / (self.state_scale + 1e-8),
                (action_batch - self.action_shift) / (self.action_scale + 1e-8),
            ],
            dim=-1,
        )
        *hidden_layers, output_layer = self._layers
        for hidden_layer in hidden_layers:
            features = torch.relu(hidden_layer(features))
        return output_layer(features)

    def predict_residual(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality), raw units
        normalized_residual_batch = self.forward(state_batch, action_batch)
        return (
            normalized_residual_batch * (self.residual_scale + 1e-8)
            + self.residual_shift
        ) * self.residual_mask

    def predict_next_state(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality), raw units
        return state_batch + self.predict_residual(
            state_batch=state_batch, action_batch=action_batch
        )


def fit_world_model(
    *,
    world_model: WorldModel,
    config: DynamicsConfig,
    state_batch: torch.Tensor,  # (transition_count, state_dimensionality)
    action_batch: torch.Tensor,  # (transition_count, action_dimensionality)
    next_state_batch: torch.Tensor,  # (transition_count, state_dimensionality)
) -> list[EpochLosses]:
    with torch.no_grad():
        world_model.initialize_transformations(
            state_batch=state_batch,
            action_batch=action_batch,
            next_state_batch=next_state_batch,
        )

    normalized_residual_targets = (
        next_state_batch - state_batch - world_model.residual_shift
    ) / (world_model.residual_scale + 1e-8)

    optimizer = torch.optim.Adam(
        params=world_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    transition_count = state_batch.shape[0]
    step_count_per_epoch = max(transition_count // config.batch_size, 1)
    epoch_losses: list[EpochLosses] = []

    world_model.train()
    for _ in tqdm(
        iterable=range(config.epoch_count),
        desc="ccil dynamics",
        unit="epoch",
        dynamic_ncols=True,
    ):
        permutation = torch.randperm(transition_count, device=state_batch.device)
        total_sum = data_sum = regularization_sum = 0.0

        for step_index in range(step_count_per_epoch):
            batch_indices = permutation[
                step_index * config.batch_size : (step_index + 1) * config.batch_size
            ]
            batch_states = state_batch[batch_indices]
            batch_actions = action_batch[batch_indices]

            optimizer.zero_grad()
            predicted_batch = world_model.forward(batch_states, batch_actions)
            target_batch = normalized_residual_targets[batch_indices]

            match config.lipschitz_type:
                case "soft_sampling":
                    data_term = torch.nn.functional.mse_loss(
                        predicted_batch, target_batch
                    )
                    regularization_term = _soft_sampling_lipschitz_penalty(
                        world_model=world_model,
                        state_batch=batch_states,
                        action_batch=batch_actions,
                        normalized_prediction_batch=predicted_batch,
                        config=config,
                    )
                case "slack":
                    if world_model.slack_network is None:
                        raise ValueError(
                            "The world model carries no slack network, so it was "
                            "built for a different regularization."
                        )
                    data_term, regularization_term = _slack_losses(
                        slack_network=world_model.slack_network,
                        state_batch=batch_states,
                        action_batch=batch_actions,
                        predicted_batch=predicted_batch,
                        target_batch=target_batch,
                        config=config,
                    )
                case "spectral_normalization" | "none":
                    data_term = torch.nn.functional.mse_loss(
                        predicted_batch, target_batch
                    )
                    regularization_term = torch.zeros((), device=data_term.device)

            loss = data_term + regularization_term
            loss.backward()
            optimizer.step()

            total_sum += float(loss.item())
            data_sum += float(data_term.item())
            regularization_sum += float(regularization_term.item())

        epoch_losses.append(
            EpochLosses(
                total=total_sum / step_count_per_epoch,
                data_term=data_sum / step_count_per_epoch,
                regularization_term=regularization_sum / step_count_per_epoch,
            )
        )

    world_model.eval()
    return epoch_losses


# ---- Private --------------------------------------------


class _SpectralNormClamp(torch.nn.Module):
    left_vector: torch.Tensor  # (output_dimensionality,)
    right_vector: torch.Tensor  # (input_dimensionality,)

    def __init__(
        self,
        *,
        input_dimensionality: int,
        output_dimensionality: int,
        lipschitz_constraint: float,
    ) -> None:
        super().__init__()
        self.lipschitz_constraint = lipschitz_constraint
        self.register_buffer(
            name="left_vector",
            tensor=torch.nn.functional.normalize(
                torch.empty(output_dimensionality).normal_(mean=0.0, std=1.0),
                dim=0,
                eps=1e-12,
            ),
        )
        self.register_buffer(
            name="right_vector",
            tensor=torch.nn.functional.normalize(
                torch.empty(input_dimensionality).normal_(mean=0.0, std=1.0),
                dim=0,
                eps=1e-12,
            ),
        )

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        left_vector = self.left_vector
        right_vector = self.right_vector

        if self.training:
            with torch.no_grad():
                torch.nn.functional.normalize(
                    torch.mv(weight.t(), left_vector),
                    dim=0,
                    eps=1e-12,
                    out=self.right_vector,
                )
                torch.nn.functional.normalize(
                    torch.mv(weight, self.right_vector),
                    dim=0,
                    eps=1e-12,
                    out=self.left_vector,
                )
            left_vector = self.left_vector.clone()
            right_vector = self.right_vector.clone()

        spectral_norm = torch.dot(left_vector, torch.mv(weight, right_vector))
        return weight / torch.clamp(spectral_norm / self.lipschitz_constraint, min=1.0)


class _SlackNetwork(torch.nn.Module):
    def __init__(
        self,
        *,
        state_dimensionality: int,
        action_dimensionality: int,
        hidden_layer_sizes: tuple[int, ...],
    ) -> None:
        super().__init__()
        layer_sizes = (
            state_dimensionality + action_dimensionality,
            *hidden_layer_sizes,
            1,
        )
        self._layers = torch.nn.ModuleList(
            modules=[
                torch.nn.Linear(
                    in_features=layer_sizes[index], out_features=layer_sizes[index + 1]
                )
                for index in range(len(layer_sizes) - 1)
            ]
        )

    def forward(
        self,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, 1)
        features = torch.cat(tensors=[state_batch, action_batch], dim=-1)
        *hidden_layers, output_layer = self._layers
        for hidden_layer in hidden_layers:
            features = torch.relu(hidden_layer(features))
        return output_layer(features) * (1.0 + 1e-8)


def _slack_losses(
    *,
    slack_network: _SlackNetwork,
    state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    predicted_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    target_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    config: DynamicsConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    slack_batch = torch.sigmoid(
        slack_network(state_batch, action_batch)
    )  # (batch_size, 1)

    error_batch = torch.nn.functional.mse_loss(
        predicted_batch, target_batch, reduction="none"
    )  # (batch_size, state_dimensionality)

    weighted_error = torch.linalg.norm(slack_batch * error_batch, dim=-1).mean()
    slack_reward = -config.slack_weight * slack_batch.mean()
    return weighted_error, slack_reward


def _soft_sampling_lipschitz_penalty(
    *,
    world_model: WorldModel,
    state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    normalized_prediction_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    config: DynamicsConfig,
) -> torch.Tensor:  # scalar
    sample_count = config.soft_sampling_sample_count

    repeated_states = state_batch.repeat_interleave(repeats=sample_count, dim=0)
    repeated_actions = action_batch.repeat_interleave(repeats=sample_count, dim=0)
    repeated_predictions = normalized_prediction_batch.repeat_interleave(
        repeats=sample_count, dim=0
    )

    state_noise = torch.randn_like(repeated_states)
    action_noise = torch.randn_like(repeated_actions)
    noisy_next_states = world_model.predict_next_state(
        state_batch=(
            repeated_states + state_noise * config.soft_sampling_standard_deviation
        ),
        action_batch=(
            repeated_actions + action_noise * config.soft_sampling_standard_deviation
        ),
    )

    noise_magnitudes = torch.linalg.norm(
        torch.cat(tensors=[state_noise, action_noise], dim=-1), dim=-1
    )
    prediction_differences = torch.linalg.norm(
        repeated_predictions - noisy_next_states, dim=-1
    )

    return (
        config.soft_sampling_penalty_weight
        * torch.relu(
            prediction_differences - config.lipschitz_constraint * noise_magnitudes
        ).mean()
    )
