import copy
from dataclasses import dataclass

import numpy as np
import torch
from numpy import typing as npt
from torch.optim.swa_utils import AveragedModel

from scripts.data.augmentation.tacil.create_dynamics.config import DynamicsConfig
from scripts.data.augmentation.tacil.create_dynamics.data import TransitionTriples
from scripts.data.augmentation.tacil.create_dynamics.loss_calculation import (
    calculate_loss,
)
from scripts.data.augmentation.tacil.dynamics.dynamics_model.dynamics_model import (
    DynamicsModel,
)
from scripts.environments.tacil_network_input import TacilNetworkInputBuilder
from scripts.environments.state_space import StateSpace
from scripts.shared.normalization import ZScoreNormalizer
from scripts.shared.seeding import set_seed

_PATIENCE_THRESHOLD = 0.001


@dataclass
class Checkpoint:
    validation_loss: float
    epoch: int
    state_dict: dict


@dataclass
class MemberEpochMetrics:
    training_loss: float
    validation_loss: float
    training_local_lipschitz_constants: npt.NDArray[np.float32]
    validation_local_lipschitz_constants: npt.NDArray[np.float32]
    gradient_norm: float


@dataclass
class MemberTrainingState:
    model: DynamicsModel
    optimizer: torch.optim.Optimizer
    ema_model: AveragedModel
    shuffle_generator: torch.Generator
    histogram_sample_generator: np.random.Generator
    state_windows: torch.Tensor
    action_windows: torch.Tensor
    best_checkpoint: Checkpoint | None = None
    patience_loss: float | None = None
    epoch_count_without_improvement: int = 0

    @property
    def ema_dynamics_model(self) -> DynamicsModel:
        module = self.ema_model.module
        if not isinstance(module, DynamicsModel):
            raise TypeError(
                f"Expected the EMA module to be a DynamicsModel, got {type(module)}"
            )
        return module


@dataclass
class TrainingContext:
    config: DynamicsConfig
    training_transitions: TransitionTriples
    validation_transitions: TransitionTriples
    validation_state_windows: (
        torch.Tensor
    )  # (window_count, window_length + 1, state_dimensionality)
    validation_action_windows: (
        torch.Tensor
    )  # (window_count, window_length, action_dimensionality)


def create_member_training_state(
    *,
    member_seed: int,
    config: DynamicsConfig,
    state_space: StateSpace,
    network_input_builder: TacilNetworkInputBuilder,
    network_input_normalizer: ZScoreNormalizer,
    state_displacement_normalizer: ZScoreNormalizer,
    state_windows: torch.Tensor,
    action_windows: torch.Tensor,
) -> MemberTrainingState:
    device = state_windows.device

    # Seeds the weight initialization below of the DynamicsModel.
    set_seed(seed=member_seed)

    dynamics_model = DynamicsModel(
        state_space=state_space,
        network_input_builder=network_input_builder,
        hidden_layer_count=config.hidden_layer_count,
        hidden_layer_dimensionality=config.hidden_layer_dimensionality,
        lipschitz_limit=config.lipschitz_limit,
        input_normalizer=network_input_normalizer,
        state_displacement_normalizer=state_displacement_normalizer,
    ).to(device=device)

    return MemberTrainingState(
        model=dynamics_model,
        optimizer=torch.optim.Adam(
            params=dynamics_model.parameters(), lr=1e-3, weight_decay=1e-5
        ),
        ema_model=AveragedModel(
            dynamics_model,
            avg_fn=lambda ema_parameter, current_parameter, _num_averaged: (
                0.9 * ema_parameter + 0.1 * current_parameter
            ),
            use_buffers=False,
        ),
        shuffle_generator=torch.Generator(device=device).manual_seed(member_seed),
        histogram_sample_generator=np.random.default_rng(seed=member_seed),
        state_windows=state_windows,
        action_windows=action_windows,
    )


def train_member_for_one_epoch(
    *, training_state: MemberTrainingState, epoch_index: int, context: TrainingContext
) -> MemberEpochMetrics:
    batch_size = 2_048

    window_count = training_state.state_windows.shape[0]
    step_count = max(window_count // batch_size, 1)

    training_state.model.train()
    shuffled_window_indices = torch.randperm(
        n=window_count,
        generator=training_state.shuffle_generator,
        device=training_state.state_windows.device,
    )

    losses: list[float] = []
    gradient_norms: list[float] = []

    for step_index in range(step_count):
        batch_window_indices = shuffled_window_indices[
            step_index * batch_size : (step_index + 1) * batch_size
        ]

        training_state.optimizer.zero_grad()

        loss = calculate_loss(
            dynamics_model=training_state.model,
            state_window_batch=training_state.state_windows[batch_window_indices],
            action_window_batch=training_state.action_windows[batch_window_indices],
        )
        loss.backward()
        losses.append(loss.detach().item())

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters=training_state.model.parameters(), max_norm=1.0
        )
        training_state.optimizer.step()
        gradient_norms.append(gradient_norm.item())

    training_state.ema_model.update_parameters(model=training_state.model)

    with torch.no_grad():
        validation_loss = _calculate_validation_loss(
            dynamics_model=training_state.ema_dynamics_model,
            state_windows=context.validation_state_windows,
            action_windows=context.validation_action_windows,
            batch_size=batch_size,
        )

    best = training_state.best_checkpoint
    if best is None or validation_loss < best.validation_loss:
        training_state.best_checkpoint = Checkpoint(
            validation_loss=validation_loss,
            epoch=epoch_index,
            state_dict=copy.deepcopy(training_state.ema_dynamics_model.state_dict()),
        )

    patience_loss = training_state.patience_loss
    if (
        patience_loss is None
        or (patience_loss - validation_loss) / patience_loss > _PATIENCE_THRESHOLD
    ):
        training_state.patience_loss = validation_loss
        training_state.epoch_count_without_improvement = 0
    else:
        training_state.epoch_count_without_improvement += 1

    with torch.no_grad():
        training_local_lipschitz_constants = _sample_local_lipschitz_constants(
            training_state=training_state,
            transitions=context.training_transitions,
            batch_size=batch_size,
        )
        validation_local_lipschitz_constants = _sample_local_lipschitz_constants(
            training_state=training_state,
            transitions=context.validation_transitions,
            batch_size=batch_size,
        )

    return MemberEpochMetrics(
        training_loss=float(np.mean(a=losses)),
        validation_loss=validation_loss,
        training_local_lipschitz_constants=training_local_lipschitz_constants,
        validation_local_lipschitz_constants=validation_local_lipschitz_constants,
        gradient_norm=float(np.max(a=gradient_norms)),
    )


# ---- Private --------------------------------------------


def _calculate_validation_loss(
    *,
    dynamics_model: DynamicsModel,
    state_windows: torch.Tensor,  # (window_count, window_length + 1, state_dimensionality)
    action_windows: torch.Tensor,  # (window_count, window_length, action_dimensionality)
    batch_size: int,
) -> float:
    window_count = state_windows.shape[0]
    total_loss = 0.0

    dynamics_model.eval()
    for batch_start_index in range(0, window_count, batch_size):
        batch_slice = slice(batch_start_index, batch_start_index + batch_size)

        state_window_batch = state_windows[
            batch_slice
        ]  # (batch_size, window_length + 1, state_dimensionality)
        action_window_batch = action_windows[
            batch_slice
        ]  # (batch_size, window_length, action_dimensionality)

        batch_loss = calculate_loss(
            dynamics_model=dynamics_model,
            state_window_batch=state_window_batch,
            action_window_batch=action_window_batch,
        )
        total_loss += batch_loss.item() * state_window_batch.shape[0]

    return total_loss / window_count


def _sample_local_lipschitz_constants(
    *,
    training_state: MemberTrainingState,
    transitions: TransitionTriples,
    batch_size: int,
) -> npt.NDArray[np.float32]:  # (sample_count,)

    transition_count = transitions.states.shape[0]
    sample_indices = training_state.histogram_sample_generator.choice(
        a=transition_count, size=min(512, transition_count), replace=False
    )  # (sample_count,)

    training_state.ema_dynamics_model.eval()

    device = next(training_state.ema_dynamics_model.parameters()).device
    states = torch.from_numpy(transitions.states[sample_indices]).to(device=device)
    actions = torch.from_numpy(transitions.actions[sample_indices]).to(device=device)

    local_lipschitz_constant_batches: list[npt.NDArray[np.float32]] = []

    for batch_start_index in range(0, states.shape[0], batch_size):
        batch_slice = slice(batch_start_index, batch_start_index + batch_size)
        local_lipschitz_constant_batches.append(
            training_state.ema_dynamics_model.calculate_local_lipschitz_constants(
                state_batch=states[batch_slice], action_batch=actions[batch_slice]
            )
            .cpu()
            .numpy()
        )
    return np.concatenate(local_lipschitz_constant_batches)
