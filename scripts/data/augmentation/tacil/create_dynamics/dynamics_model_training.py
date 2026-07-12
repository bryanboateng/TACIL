import copy
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from numpy import typing as npt
from torch.optim.swa_utils import AveragedModel
from tqdm import tqdm
from wandb import Run

from scripts.data.augmentation.tacil.create_dynamics.config import Config
from scripts.data.augmentation.tacil.create_dynamics.data import TransitionSet
from scripts.data.augmentation.tacil.create_dynamics.evaluation import (
    calculate_local_lipschitz_constants,
    calculate_validation_loss,
)
from scripts.data.augmentation.tacil.create_dynamics.losses import compute_loss
from scripts.data.augmentation.tacil.models.dynamics_model import DynamicsModel
from scripts.data.augmentation.tacil.models.normalizers import Normalizers


@dataclass
class Checkpoint:
    validation_loss: float
    epoch: int
    state_dict: dict


@dataclass
class LocalLipschitzEpochHistory:
    """One member's per-epoch local Lipschitz statistics on one data split."""

    means: list[float]
    p99s: list[float]
    # per-epoch subsample of local Lipschitz constants; the ensemble script pools these
    # across members into one histogram per epoch
    subsamples: list[npt.NDArray[np.float32]]


@dataclass
class MemberEpochHistory:
    """One ensemble member's per-epoch evaluation record."""

    validation_losses: list[float]
    training_lipschitz: LocalLipschitzEpochHistory
    validation_lipschitz: LocalLipschitzEpochHistory


def train_dynamics_model(
    *,
    member_index: int,
    config: Config,
    normalizers: Normalizers,
    training_transitions: TransitionSet,
    validation_transitions: TransitionSet,
    train_state_windows: npt.NDArray[np.float32],
    train_action_windows: npt.NDArray[np.float32],
    validation_state_windows: npt.NDArray[np.float32],
    validation_action_windows: npt.NDArray[np.float32],
    run: Run,
    device: torch.device,
    initial_global_step: int,
) -> tuple[DynamicsModel, int, MemberEpochHistory]:
    dynamics_model = DynamicsModel(
        state_dimensionality=training_transitions.states.shape[1],
        action_dimensionality=training_transitions.actions.shape[1],
        hidden_dimensionalities=config.architecture.hidden_layer_sizes,
        lipschitz_constant_upper_bound=config.architecture.lipschitz_constant_upper_bound,
    ).to(device=device)

    optimizer = torch.optim.Adam(
        params=dynamics_model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    ema_model = AveragedModel(
        dynamics_model,
        avg_fn=lambda ema_parameter, current_parameter, _num_averaged: (
            config.training.ema_decay * ema_parameter
            + (1.0 - config.training.ema_decay) * current_parameter
        ),
        use_buffers=False,
    )

    # AveragedModel.module is typed as nn.Module but it is the DynamicsNetwork we
    # wrapped.
    ema_dynamics_model = cast(DynamicsModel, ema_model.module)

    train_state_windows_tensor = torch.from_numpy(ndarray=train_state_windows).to(
        device=device
    )
    train_action_windows_tensor = torch.from_numpy(ndarray=train_action_windows).to(
        device=device
    )

    window_count = train_state_windows_tensor.shape[0]
    steps_per_epoch = max(window_count // config.data.loader.batch_size, 1)

    global_step = initial_global_step
    best_checkpoint: Checkpoint | None = None
    validation_losses: list[float] = []
    training_lipschitz = LocalLipschitzEpochHistory(means=[], p99s=[], subsamples=[])
    validation_lipschitz = LocalLipschitzEpochHistory(means=[], p99s=[], subsamples=[])
    histogram_sample_rng = np.random.default_rng(seed=member_index)

    for epoch_index in tqdm(
        range(config.training.epoch_count),
        desc=f"training member {member_index}",
        unit="epoch",
    ):
        dynamics_model.train()

        shuffled_window_indices = torch.randperm(n=window_count, device=device)
        for step_index in range(steps_per_epoch):
            batch_window_indices = shuffled_window_indices[
                step_index * config.data.loader.batch_size : (step_index + 1)
                * config.data.loader.batch_size
            ]

            optimizer.zero_grad()

            loss = compute_loss(
                dynamics_model=dynamics_model,
                normalizers=normalizers,
                state_window_batch=train_state_windows_tensor[batch_window_indices],
                action_window_batch=train_action_windows_tensor[batch_window_indices],
            )
            loss.backward()

            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters=dynamics_model.parameters(), max_norm=1.0
            )
            optimizer.step()

            # These per-step training-health signals share one panel across all
            # members (no member prefix): the members occupy consecutive global-step
            # ranges, so this reads as one continuous curve where a member that fails
            # to descend or blows up still stands out. Per-member detail lives in the
            # validation-loss and Lipschitz curves instead.
            run.log(
                {
                    "loss/training": loss.detach().item(),
                    "gradient_norm": gradient_norm.item(),
                },
                step=global_step,
            )
            global_step += 1

        ema_model.update_parameters(model=dynamics_model)

        if config.architecture.lipschitz_constant_upper_bound is not None:
            _refresh_spectral_norm_estimates(ema_dynamics_model=ema_dynamics_model)

        validation_loss = calculate_validation_loss(
            dynamics_model=ema_dynamics_model,
            normalizers=normalizers,
            state_windows=validation_state_windows,
            action_windows=validation_action_windows,
            batch_size=config.data.loader.batch_size,
        )
        # Per-member curves are not logged: with a large ensemble they overwhelm the
        # dashboard. Each member's per-epoch statistics are collected here instead, and
        # the ensemble script logs the across-member aggregate (mean plus a
        # standard-deviation band and a pooled histogram), which already surfaces a
        # single misbehaving member.
        validation_losses.append(validation_loss)
        for lipschitz_epoch_history, split_states, split_actions in (
            (
                training_lipschitz,
                training_transitions.states,
                training_transitions.actions,
            ),
            (
                validation_lipschitz,
                validation_transitions.states,
                validation_transitions.actions,
            ),
        ):
            local_lipschitz_constants = calculate_local_lipschitz_constants(
                dynamics_model=ema_dynamics_model,
                normalizers=normalizers,
                states=split_states,
                actions=split_actions,
                batch_size=config.data.loader.batch_size,
            )
            lipschitz_epoch_history.means.append(
                float(np.mean(local_lipschitz_constants))
            )
            lipschitz_epoch_history.p99s.append(
                float(np.percentile(local_lipschitz_constants, q=99.0))
            )
            # Keep a bounded subsample so the ensemble script can pool constants across
            # members into one histogram per epoch without holding every constant.
            lipschitz_epoch_history.subsamples.append(
                _subsample_for_histogram(
                    values=local_lipschitz_constants, rng=histogram_sample_rng
                )
            )

        if best_checkpoint is None or validation_loss < best_checkpoint.validation_loss:
            best_checkpoint = Checkpoint(
                validation_loss=validation_loss,
                epoch=epoch_index,
                state_dict=copy.deepcopy(ema_dynamics_model.state_dict()),
            )

    if best_checkpoint is None:
        raise ValueError("No checkpoint was recorded")

    dynamics_model.load_state_dict(state_dict=best_checkpoint.state_dict)

    return (
        dynamics_model,
        global_step,
        MemberEpochHistory(
            validation_losses=validation_losses,
            training_lipschitz=training_lipschitz,
            validation_lipschitz=validation_lipschitz,
        ),
    )


def _subsample_for_histogram(
    *, values: npt.NDArray[np.float32], rng: np.random.Generator
) -> npt.NDArray[np.float32]:
    maximum_sample_count = 512
    if values.shape[0] <= maximum_sample_count:
        return values
    selected_indices = rng.choice(
        a=values.shape[0], size=maximum_sample_count, replace=False
    )
    return values[selected_indices]


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
