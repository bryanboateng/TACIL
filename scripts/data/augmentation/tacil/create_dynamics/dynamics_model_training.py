import copy
from dataclasses import dataclass

import numpy as np
import torch
from numpy import typing as npt

from scripts.data.augmentation.tacil.create_dynamics.loss_calculation import (
    calculate_losses,
)
from scripts.data.augmentation.tacil.dynamics.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)

_BATCH_SIZE = 2_048
_LIPSCHITZ_SAMPLE_COUNT = 512
_IMPROVEMENT_THRESHOLD = 0.01


@dataclass
class EpochMetrics:
    training_losses: npt.NDArray[np.float64]  # (member_count,)
    validation_losses: npt.NDArray[np.float64]  # (member_count,)
    gradient_norms: npt.NDArray[np.float64]  # (member_count,)
    training_local_lipschitz_constants: npt.NDArray[
        np.float32
    ]  # (member_count, sample_count)
    validation_local_lipschitz_constants: npt.NDArray[
        np.float32
    ]  # (member_count, sample_count)


@dataclass
class TrainingState:
    model: DynamicsModelEnsemble
    best_model: DynamicsModelEnsemble
    optimizer: torch.optim.Optimizer
    generator: torch.Generator
    window_indices_per_member: torch.Tensor  # (member_count, window_count)
    best_validation_losses: npt.NDArray[np.float64]  # (member_count,)
    best_epochs: npt.NDArray[np.int64]  # (member_count,)
    last_improvement_epoch: int


@dataclass
class TrainingContext:
    training_state_windows: (
        torch.Tensor
    )  # (window_count, window_length + 1, state_dimensionality)
    training_action_windows: (
        torch.Tensor
    )  # (window_count, window_length, action_dimensionality)
    validation_state_windows: (
        torch.Tensor
    )  # (window_count, window_length + 1, state_dimensionality)
    validation_action_windows: (
        torch.Tensor
    )  # (window_count, window_length, action_dimensionality)


def create_training_state(
    *, model: DynamicsModelEnsemble, seed: int, training_window_count: int
) -> TrainingState:
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)

    return TrainingState(
        model=model,
        best_model=copy.deepcopy(model),
        optimizer=torch.optim.Adam(
            params=model.parameters(), lr=1e-3, weight_decay=1e-5
        ),
        generator=generator,
        window_indices_per_member=torch.randint(
            low=0,
            high=training_window_count,
            size=(model.member_count, training_window_count),
            generator=generator,
            device=device,
        ),
        best_validation_losses=np.full(shape=model.member_count, fill_value=np.inf),
        best_epochs=np.full(shape=model.member_count, fill_value=-1, dtype=np.int64),
        last_improvement_epoch=-1,
    )


def train_for_one_epoch(
    *, training_state: TrainingState, epoch_index: int, context: TrainingContext
) -> EpochMetrics:
    device = context.training_state_windows.device
    member_count = training_state.model.member_count

    window_indices_per_member = _shuffle_rows(
        index_per_member=training_state.window_indices_per_member,
        generator=training_state.generator,
    )  # (member_count, window_count)
    step_count = max(window_indices_per_member.shape[1] // _BATCH_SIZE, 1)

    training_state.model.train()

    loss_sums = torch.zeros(member_count, dtype=torch.float64, device=device)
    maximum_gradient_norms = torch.zeros(member_count, device=device)

    for step_index in range(step_count):
        batch_window_indices = window_indices_per_member[
            :, step_index * _BATCH_SIZE : (step_index + 1) * _BATCH_SIZE
        ]  # (member_count, batch_size)

        training_state.optimizer.zero_grad()

        losses = calculate_losses(
            dynamics_model_ensemble=training_state.model,
            state_window_batch=context.training_state_windows[batch_window_indices],
            action_window_batch=context.training_action_windows[batch_window_indices],
        )  # (member_count,)
        losses.sum().backward()
        loss_sums += losses.detach()

        gradient_norms = _clip_gradient_norms_per_member(
            model=training_state.model, maximum_norm=1.0
        )  # (member_count,)
        training_state.optimizer.step()
        maximum_gradient_norms = torch.maximum(maximum_gradient_norms, gradient_norms)

    with torch.no_grad():
        validation_losses = (
            _calculate_validation_losses(
                dynamics_model_ensemble=training_state.model,
                state_windows=context.validation_state_windows,
                action_windows=context.validation_action_windows,
            )
            .cpu()
            .numpy()
        )  # (member_count,)

        _record_checkpoints(
            training_state=training_state,
            validation_losses=validation_losses,
            epoch_index=epoch_index,
        )

        training_local_lipschitz_constants = _sample_local_lipschitz_constants(
            training_state=training_state,
            state_windows=context.training_state_windows,
            action_windows=context.training_action_windows,
        )  # (member_count, sample_count)
        validation_local_lipschitz_constants = _sample_local_lipschitz_constants(
            training_state=training_state,
            state_windows=context.validation_state_windows,
            action_windows=context.validation_action_windows,
        )  # (member_count, sample_count)

    return EpochMetrics(
        training_losses=(loss_sums / step_count).cpu().numpy(),
        validation_losses=validation_losses,
        gradient_norms=maximum_gradient_norms.double().cpu().numpy(),
        training_local_lipschitz_constants=training_local_lipschitz_constants,
        validation_local_lipschitz_constants=validation_local_lipschitz_constants,
    )


# ---- Private --------------------------------------------


def _shuffle_rows(
    *,
    index_per_member: torch.Tensor,  # (member_count, window_count)
    generator: torch.Generator,
) -> torch.Tensor:  # (member_count, window_count)
    order = torch.rand(
        index_per_member.shape, generator=generator, device=index_per_member.device
    ).argsort(dim=-1)
    return index_per_member.gather(dim=-1, index=order)


def _clip_gradient_norms_per_member(
    *, model: DynamicsModelEnsemble, maximum_norm: float
) -> torch.Tensor:  # (member_count,)
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]  # each (member_count, ...)

    total_norms = (
        torch.stack(
            tensors=[
                gradient.flatten(start_dim=1).square().sum(dim=1)
                for gradient in gradients
            ],
            dim=0,
        )
        .sum(dim=0)
        .sqrt()
    )  # (member_count,)

    clip_coefficients = torch.clamp(
        input=maximum_norm / (total_norms + 1e-6), max=1.0
    )  # (member_count,)
    for gradient in gradients:
        gradient.mul_(clip_coefficients.view(-1, *(1,) * (gradient.dim() - 1)))

    return total_norms


def _calculate_validation_losses(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    state_windows: torch.Tensor,  # (window_count, window_length + 1, state_dimensionality)
    action_windows: torch.Tensor,  # (window_count, window_length, action_dimensionality)
) -> torch.Tensor:  # (member_count,)
    dynamics_model_ensemble.eval()
    weighted_losses = torch.stack(
        tensors=[
            calculate_losses(
                dynamics_model_ensemble=dynamics_model_ensemble,
                state_window_batch=state_window_batch,
                action_window_batch=action_window_batch,
            )
            * state_window_batch.shape[0]
            for state_window_batch, action_window_batch in zip(
                state_windows.split(_BATCH_SIZE),
                action_windows.split(_BATCH_SIZE),
                strict=True,
            )
        ],
        dim=0,
    )  # (batch_count, member_count)
    return weighted_losses.sum(dim=0) / state_windows.shape[0]


def _record_checkpoints(
    *,
    training_state: TrainingState,
    validation_losses: npt.NDArray[np.float64],  # (member_count,)
    epoch_index: int,
) -> None:
    improved_member_mask = validation_losses < training_state.best_validation_losses * (
        1.0 - _IMPROVEMENT_THRESHOLD
    )  # (member_count,)
    training_state.best_validation_losses[improved_member_mask] = validation_losses[
        improved_member_mask
    ]
    training_state.best_epochs[improved_member_mask] = epoch_index
    if improved_member_mask.any():
        training_state.last_improvement_epoch = epoch_index

    improved_member_indices = np.flatnonzero(
        improved_member_mask
    ).tolist()  # (improved_member_count,)
    for best_parameter, parameter in zip(
        training_state.best_model.parameters(),
        training_state.model.parameters(),
        strict=True,
    ):
        best_parameter[improved_member_indices] = parameter[improved_member_indices]


def _sample_local_lipschitz_constants(
    *,
    training_state: TrainingState,
    state_windows: torch.Tensor,  # (window_count, window_length + 1, state_dimensionality)
    action_windows: torch.Tensor,  # (window_count, window_length, action_dimensionality)
) -> npt.NDArray[np.float32]:  # (member_count, sample_count)
    sample_indices = torch.randint(
        low=0,
        high=state_windows.shape[0],
        size=(training_state.model.member_count, _LIPSCHITZ_SAMPLE_COUNT),
        generator=training_state.generator,
        device=state_windows.device,
    )  # (member_count, sample_count)

    return (
        training_state.model.calculate_local_lipschitz_constants(
            state_batch=state_windows[sample_indices, 0],
            action_batch=action_windows[sample_indices, 0],
        )
        .cpu()
        .numpy()
    )
