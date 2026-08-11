import copy
from dataclasses import dataclass

import numpy as np
import torch
from numpy import typing as npt
from torch.optim.swa_utils import AveragedModel

from scripts.data.augmentation.tacil.create_dynamics.config import DynamicsConfig
from scripts.data.augmentation.tacil.create_dynamics.data import TransitionSet
from scripts.data.augmentation.tacil.create_dynamics.evaluation import (
    calculate_local_lipschitz_constants,
    calculate_validation_loss,
)
from scripts.data.augmentation.tacil.create_dynamics.losses import compute_loss
from scripts.data.augmentation.tacil.models.dynamics_model import DynamicsModel
from scripts.data.augmentation.tacil.models.normalizers import Normalizers
from scripts.shared.seeding import set_seed


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
    best_checkpoint: Checkpoint | None = None

    @property
    def ema_dynamics_model(self) -> DynamicsModel:
        # AveragedModel.module is typed as nn.Module; verify at runtime that it is the
        # DynamicsModel we wrapped rather than trusting the annotation.
        module = self.ema_model.module
        if not isinstance(module, DynamicsModel):
            raise TypeError(
                f"Expected the EMA module to be a DynamicsModel, got {type(module)}"
            )
        return module


@dataclass
class TrainingContext:
    config: DynamicsConfig
    normalizers: Normalizers
    training_transitions: TransitionSet
    validation_transitions: TransitionSet
    train_state_windows: torch.Tensor
    train_action_windows: torch.Tensor
    validation_state_windows: npt.NDArray[np.float32]
    validation_action_windows: npt.NDArray[np.float32]


def create_member_training_state(
    *,
    member_seed: int,
    config: DynamicsConfig,
    transitions: TransitionSet,
    device: torch.device,
) -> MemberTrainingState:
    set_seed(seed=member_seed)

    dynamics_model = DynamicsModel(
        state_dimensionality=transitions.states.shape[1],
        action_dimensionality=transitions.actions.shape[1],
        hidden_layer_count=config.hidden_layer_count,
        hidden_layer_dimensionality=config.hidden_layer_dimensionality,
        lipschitz_constant_upper_bound=config.lipschitz_constant_upper_bound,
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
    )


def train_member_for_one_epoch(
    *, state: MemberTrainingState, epoch_index: int, context: TrainingContext
) -> MemberEpochMetrics:
    batch_size = 2_048

    window_count = context.train_state_windows.shape[0]
    steps_per_epoch = max(window_count // batch_size, 1)

    state.model.train()
    shuffled_window_indices = torch.randperm(
        n=window_count,
        generator=state.shuffle_generator,
        device=context.train_state_windows.device,
    )

    losses: list[float] = []
    gradient_norms: list[float] = []

    for step_index in range(steps_per_epoch):
        batch_window_indices = shuffled_window_indices[
            step_index * batch_size : (step_index + 1) * batch_size
        ]

        state.optimizer.zero_grad()

        loss = compute_loss(
            dynamics_model=state.model,
            normalizers=context.normalizers,
            state_window_batch=context.train_state_windows[batch_window_indices],
            action_window_batch=context.train_action_windows[batch_window_indices],
        )
        loss.backward()
        losses.append(loss.detach().item())

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters=state.model.parameters(), max_norm=1.0
        )
        state.optimizer.step()
        gradient_norms.append(gradient_norm.item())

    state.ema_model.update_parameters(model=state.model)

    validation_loss = calculate_validation_loss(
        dynamics_model=state.ema_dynamics_model,
        normalizers=context.normalizers,
        state_windows=context.validation_state_windows,
        action_windows=context.validation_action_windows,
        batch_size=batch_size,
    )

    if (
        state.best_checkpoint is None
        or validation_loss < state.best_checkpoint.validation_loss
    ):
        state.best_checkpoint = Checkpoint(
            validation_loss=validation_loss,
            epoch=epoch_index,
            state_dict=copy.deepcopy(state.ema_dynamics_model.state_dict()),
        )

    return MemberEpochMetrics(
        training_loss=float(np.mean(a=losses)),
        validation_loss=validation_loss,
        training_local_lipschitz_constants=_sample_local_lipschitz_constants(
            state=state,
            normalizers=context.normalizers,
            transitions=context.training_transitions,
            batch_size=batch_size,
        ),
        validation_local_lipschitz_constants=_sample_local_lipschitz_constants(
            state=state,
            normalizers=context.normalizers,
            transitions=context.validation_transitions,
            batch_size=batch_size,
        ),
        gradient_norm=float(np.max(a=gradient_norms)),
    )


def load_best_member_model(
    *, member_training_state: MemberTrainingState
) -> DynamicsModel:
    if member_training_state.best_checkpoint is None:
        raise ValueError("No checkpoint was recorded")
    member_training_state.model.load_state_dict(
        state_dict=member_training_state.best_checkpoint.state_dict
    )
    return member_training_state.model


# ---- Private --------------------------------------------


def _sample_local_lipschitz_constants(
    *,
    state: MemberTrainingState,
    normalizers: Normalizers,
    transitions: TransitionSet,
    batch_size: int,
) -> npt.NDArray[np.float32]:  # (sample_count,)
    transition_count = transitions.states.shape[0]
    sample_indices = state.histogram_sample_generator.choice(
        a=transition_count, size=min(512, transition_count), replace=False
    )  # (sample_count,)
    return calculate_local_lipschitz_constants(
        dynamics_model=state.ema_dynamics_model,
        normalizers=normalizers,
        states=transitions.states[sample_indices],
        actions=transitions.actions[sample_indices],
        batch_size=batch_size,
    )
