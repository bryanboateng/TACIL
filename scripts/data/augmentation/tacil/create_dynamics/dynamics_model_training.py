import copy
from dataclasses import dataclass

import numpy as np
import torch
from numpy import typing as npt
from torch.optim.swa_utils import AveragedModel

from scripts.data.augmentation.tacil.create_dynamics.config import Config
from scripts.data.augmentation.tacil.create_dynamics.data import TransitionSet
from scripts.data.augmentation.tacil.create_dynamics.evaluation import (
    calculate_local_lipschitz_constants,
    calculate_validation_loss,
)
from scripts.data.augmentation.tacil.create_dynamics.losses import compute_loss
from scripts.data.augmentation.tacil.models.dynamics_model import DynamicsModel
from scripts.data.augmentation.tacil.models.normalizers import Normalizers
from scripts.shared.seed import set_seed


@dataclass
class Checkpoint:
    validation_loss: float
    epoch: int
    state_dict: dict


@dataclass
class MemberEpochMetrics:
    training_loss: float
    gradient_norm: float
    validation_loss: float
    training_local_lipschitz_constants: npt.NDArray[np.float32]
    validation_local_lipschitz_constants: npt.NDArray[np.float32]


@dataclass
class MemberTrainingState:
    model: DynamicsModel
    optimizer: torch.optim.Optimizer
    ema_model: AveragedModel
    shuffle_generator: torch.Generator
    histogram_sample_rng: np.random.Generator
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
    config: Config
    normalizers: Normalizers
    training_transitions: TransitionSet
    validation_transitions: TransitionSet
    train_state_windows: torch.Tensor
    train_action_windows: torch.Tensor
    validation_state_windows: npt.NDArray[np.float32]
    validation_action_windows: npt.NDArray[np.float32]


def create_member_training_state(
    *,
    member_index: int,
    member_seed: int,
    config: Config,
    transitions: TransitionSet,
    device: torch.device,
) -> MemberTrainingState:
    set_seed(seed=member_seed)

    dynamics_model = DynamicsModel(
        state_dimensionality=transitions.states.shape[1],
        action_dimensionality=transitions.actions.shape[1],
        hidden_dimensionalities=config.architecture.hidden_layer_sizes,
        lipschitz_constant_upper_bound=config.architecture.lipschitz_constant_upper_bound,
    ).to(device=device)

    ema_model = AveragedModel(
        dynamics_model,
        avg_fn=lambda ema_parameter, current_parameter, _num_averaged: (
            config.training.ema_decay * ema_parameter
            + (1.0 - config.training.ema_decay) * current_parameter
        ),
        use_buffers=False,
    )

    return MemberTrainingState(
        model=dynamics_model,
        optimizer=torch.optim.Adam(
            params=dynamics_model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        ),
        ema_model=ema_model,
        shuffle_generator=torch.Generator(device=device).manual_seed(seed=member_seed),
        histogram_sample_rng=np.random.default_rng(seed=member_index),
    )


def train_member_for_one_epoch(
    *, state: MemberTrainingState, epoch_index: int, context: TrainingContext
) -> MemberEpochMetrics:
    config = context.config
    window_count = context.train_state_windows.shape[0]
    steps_per_epoch = max(window_count // config.data.loader.batch_size, 1)

    state.model.train()
    shuffled_window_indices = torch.randperm(
        n=window_count,
        generator=state.shuffle_generator,
        device=context.train_state_windows.device,
    )

    step_losses: list[float] = []
    step_gradient_norms: list[float] = []
    for step_index in range(steps_per_epoch):
        batch_window_indices = shuffled_window_indices[
            step_index * config.data.loader.batch_size : (step_index + 1)
            * config.data.loader.batch_size
        ]

        state.optimizer.zero_grad()
        loss = compute_loss(
            dynamics_model=state.model,
            normalizers=context.normalizers,
            state_window_batch=context.train_state_windows[batch_window_indices],
            action_window_batch=context.train_action_windows[batch_window_indices],
        )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters=state.model.parameters(), max_norm=1.0
        )
        state.optimizer.step()

        step_losses.append(loss.detach().item())
        step_gradient_norms.append(gradient_norm.item())

    state.ema_model.update_parameters(model=state.model)
    if config.architecture.lipschitz_constant_upper_bound is not None:
        _refresh_spectral_norm_estimates(ema_dynamics_model=state.ema_dynamics_model)

    validation_loss = calculate_validation_loss(
        dynamics_model=state.ema_dynamics_model,
        normalizers=context.normalizers,
        state_windows=context.validation_state_windows,
        action_windows=context.validation_action_windows,
        batch_size=config.data.loader.batch_size,
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
        # Mean loss over the epoch's steps and the epoch's worst gradient norm: a spike
        # in the latter is the earliest sign a member is going unstable.
        training_loss=float(np.mean(step_losses)),
        gradient_norm=float(np.max(step_gradient_norms)),
        validation_loss=validation_loss,
        training_local_lipschitz_constants=_compute_split_lipschitz(
            ema_dynamics_model=state.ema_dynamics_model,
            normalizers=context.normalizers,
            transitions=context.training_transitions,
            batch_size=config.data.loader.batch_size,
            histogram_sample_rng=state.histogram_sample_rng,
        ),
        validation_local_lipschitz_constants=_compute_split_lipschitz(
            ema_dynamics_model=state.ema_dynamics_model,
            normalizers=context.normalizers,
            transitions=context.validation_transitions,
            batch_size=config.data.loader.batch_size,
            histogram_sample_rng=state.histogram_sample_rng,
        ),
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


def _compute_split_lipschitz(
    *,
    ema_dynamics_model: DynamicsModel,
    normalizers: Normalizers,
    transitions: TransitionSet,
    batch_size: int,
    histogram_sample_rng: np.random.Generator,
) -> npt.NDArray[np.float32]:
    local_lipschitz_constants = calculate_local_lipschitz_constants(
        dynamics_model=ema_dynamics_model,
        normalizers=normalizers,
        states=transitions.states,
        actions=transitions.actions,
        batch_size=batch_size,
    )
    # A bounded subsample is enough for the pooled histogram and keeps memory flat.
    maximum_sample_count = 512
    if local_lipschitz_constants.shape[0] <= maximum_sample_count:
        return local_lipschitz_constants
    selected_indices = histogram_sample_rng.choice(
        a=local_lipschitz_constants.shape[0], size=maximum_sample_count, replace=False
    )
    return local_lipschitz_constants[selected_indices]


@torch.no_grad()
def _refresh_spectral_norm_estimates(*, ema_dynamics_model: DynamicsModel) -> None:
    """Recompute the spectral-norm vectors for the EMA network's averaged weights.

    These vectors only update during train-mode forward passes, and the EMA network is
    never trained, so its vectors are out of date. They also can't be copied from the
    live network, because they only fit the exact weights they were computed from, and
    the EMA network has different weights. So we run a few forward
    passes in train mode: each pass moves the vectors one step closer to the correct
    values for the current weights.
    The input values play no role here, so we use zeros.
    """
    network_device = next(ema_dynamics_model.parameters()).device

    dummy_normalized_input_batch = torch.zeros(
        1,
        ema_dynamics_model.state_dimensionality
        + ema_dynamics_model.action_dimensionality,
        device=network_device,
    )

    model_was_training = ema_dynamics_model.training
    ema_dynamics_model.train()
    for _ in range(50):
        ema_dynamics_model(dummy_normalized_input_batch)
    ema_dynamics_model.train(mode=model_was_training)
