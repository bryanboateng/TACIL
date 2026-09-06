from dataclasses import dataclass

import torch
from tqdm import tqdm

from scripts.data.augmentation.ccil.config import DynamicsConfig, LipschitzType


@dataclass(frozen=True)
class EpochLosses:
    total: float
    mse: float
    lipschitz_penalty: float


class _SpectralNormClamp(torch.nn.Module):
    def __init__(self, *, lipschitz_constraint: float) -> None:
        super().__init__()
        self.lipschitz_constraint = lipschitz_constraint

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        spectral_norm = torch.linalg.matrix_norm(weight, ord=2)
        return weight / torch.clamp(spectral_norm / self.lipschitz_constraint, min=1.0)


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
        if lipschitz_type == "spectral_normalization":
            for layer in self._layers:
                torch.nn.utils.parametrize.register_parametrization(
                    layer,
                    "weight",
                    _SpectralNormClamp(lipschitz_constraint=lipschitz_constraint),
                )

        # Transformation buffers, overwritten by initialize_transformations.
        self.register_buffer("state_shift", torch.zeros(state_dimensionality))
        self.register_buffer("state_scale", torch.ones(state_dimensionality))
        self.register_buffer("action_shift", torch.zeros(action_dimensionality))
        self.register_buffer("action_scale", torch.ones(action_dimensionality))
        self.register_buffer("residual_shift", torch.zeros(state_dimensionality))
        self.register_buffer("residual_scale", torch.ones(state_dimensionality))
        self.register_buffer("residual_mask", torch.ones(state_dimensionality))

    @torch.no_grad()
    def initialize_transformations(
        self,
        *,
        state_batch: torch.Tensor,  # (transition_count, state_dimensionality)
        action_batch: torch.Tensor,  # (transition_count, action_dimensionality)
        next_state_batch: torch.Tensor,  # (transition_count, state_dimensionality)
    ) -> None:
        """Mean shift, mean-absolute-deviation scale — exactly their
        init_normalization_to_data."""
        # Plain subtraction on the angle too. See the module docstring.
        residual_batch = next_state_batch - state_batch

        self.state_shift = state_batch.mean(dim=0)
        self.state_scale = (state_batch - self.state_shift).abs().mean(dim=0)
        self.action_shift = action_batch.mean(dim=0)
        self.action_scale = (action_batch - self.action_shift).abs().mean(dim=0)
        self.residual_shift = residual_batch.mean(dim=0)
        self.residual_scale = (residual_batch - self.residual_shift).abs().mean(dim=0)
        # Dimensions that barely move are forced to a zero prediction.
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
        """Their model.f: raw (state, action) -> raw state residual."""
        normalized_residual_batch = self.forward(state_batch, action_batch)
        return (
            normalized_residual_batch * (self.residual_scale + 1e-8)
            + self.residual_shift
        ) * self.residual_mask


def fit_world_model(
    *,
    world_model: WorldModel,
    config: DynamicsConfig,
    state_batch: torch.Tensor,  # (transition_count, state_dimensionality)
    action_batch: torch.Tensor,  # (transition_count, action_dimensionality)
    next_state_batch: torch.Tensor,  # (transition_count, state_dimensionality)
) -> list[EpochLosses]:
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
        iterable=range(config.epoch_count), desc="ccil dynamics", unit="epoch"
    ):
        permutation = torch.randperm(transition_count, device=state_batch.device)
        total_sum = mse_sum = penalty_sum = 0.0

        for step_index in range(step_count_per_epoch):
            batch_indices = permutation[
                step_index * config.batch_size : (step_index + 1) * config.batch_size
            ]
            batch_states = state_batch[batch_indices]
            batch_actions = action_batch[batch_indices]

            optimizer.zero_grad()
            predicted_batch = world_model.forward(batch_states, batch_actions)
            mse_loss = torch.nn.functional.mse_loss(
                predicted_batch, normalized_residual_targets[batch_indices]
            )
            if config.lipschitz_type == "soft_sampling":
                lipschitz_penalty = _soft_sampling_lipschitz_penalty(
                    world_model=world_model,
                    state_batch=batch_states,
                    action_batch=batch_actions,
                    predicted_residual_batch=world_model.predict_residual(
                        state_batch=batch_states, action_batch=batch_actions
                    ),
                    config=config,
                )
            else:
                # spectral_normalization constrains through the weight
                # parametrization; "none" is plain MSE.
                lipschitz_penalty = torch.zeros((), device=mse_loss.device)
            loss = mse_loss + lipschitz_penalty
            loss.backward()
            optimizer.step()

            total_sum += float(loss.item())
            mse_sum += float(mse_loss.item())
            penalty_sum += float(lipschitz_penalty.item())

        epoch_losses.append(
            EpochLosses(
                total=total_sum / step_count_per_epoch,
                mse=mse_sum / step_count_per_epoch,
                lipschitz_penalty=penalty_sum / step_count_per_epoch,
            )
        )

    world_model.eval()
    return epoch_losses


# ---- Private --------------------------------------------


def _soft_sampling_lipschitz_penalty(
    *,
    world_model: WorldModel,
    state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    predicted_residual_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    config: DynamicsConfig,
) -> torch.Tensor:  # scalar
    sample_count = config.soft_lipschitz_sample_count

    repeated_states = state_batch.repeat_interleave(repeats=sample_count, dim=0)
    repeated_actions = action_batch.repeat_interleave(repeats=sample_count, dim=0)
    repeated_predictions = predicted_residual_batch.repeat_interleave(
        repeats=sample_count, dim=0
    )

    state_noise = torch.randn_like(repeated_states)
    action_noise = torch.randn_like(repeated_actions)
    noisy_predictions = world_model.predict_residual(
        state_batch=repeated_states + state_noise * config.soft_lipschitz_sampling_eps,
        action_batch=(
            repeated_actions + action_noise * config.soft_lipschitz_sampling_eps
        ),
    )

    noise_magnitudes = torch.linalg.norm(
        torch.cat(tensors=[state_noise, action_noise], dim=-1), dim=-1
    )
    prediction_differences = torch.linalg.norm(
        repeated_predictions - noisy_predictions, dim=-1
    )

    return (
        config.soft_lipschitz_penalty_weight
        * torch.relu(
            prediction_differences - config.lipschitz_constraint * noise_magnitudes
        ).mean()
    )
