import copy
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import cast

import numpy as np
import torch
import tyro
import wandb
from numpy import typing as npt
from rich.pretty import pretty_repr
from sklearn.model_selection import train_test_split
from torch.optim.swa_utils import AveragedModel
from tqdm import tqdm
from wandb import Run

from scripts.ashared import (
    compute_config_hash,
    create_logger,
    log_device_info,
    set_seed,
)
from scripts.data.augmentation.ccil.dynamics.checkpoint import save_dynamics_checkpoint
from scripts.data.augmentation.ccil.dynamics.config import Config
from scripts.data.augmentation.ccil.dynamics.data import (
    build_windows,
    concatenate_episodes,
    load_episodes,
)
from scripts.data.augmentation.ccil.dynamics.evaluation import (
    calculate_local_lipschitz_constants,
    calculate_validation_loss,
)
from scripts.data.augmentation.ccil.dynamics.losses import compute_loss
from scripts.data.augmentation.ccil.dynamics_model import DynamicsModel
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH

logger = create_logger()


def main():
    config = tyro.cli(f=Config, config=[tyro.conf.UsePythonSyntaxForLiteralCollections])
    logger.info(pretty_repr(config))
    _run(config=config)


def _run(config: Config) -> None:
    set_seed(seed=config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_device_info(device=device, logger=logger)

    with wandb.init(
        project="abc",
        job_type="train",
        config=asdict(config),
        dir=OUTPUT_DIRECTORY_PATH,
    ) as run:
        training_episodes, validation_episodes = train_test_split(
            load_episodes(
                artifact=run.use_artifact(artifact_or_name=config.data.artifact),
                logger=logger,
            ),
            test_size=config.data.split.validation_fraction,
            random_state=config.data.split.seed,
            shuffle=True,
        )
        logger.info("# Training episodes: %d", len(training_episodes))
        logger.info("# Validation episodes: %d", len(validation_episodes))

        train_transitions = concatenate_episodes(episodes=training_episodes)
        validation_transitions = concatenate_episodes(episodes=validation_episodes)

        dynamics_model = DynamicsModel(
            state_dimensionality=train_transitions.states.shape[1],
            action_dimensionality=train_transitions.actions.shape[1],
            hidden_dimensionalities=config.architecture.hidden_layer_sizes,
            lipschitz_constant_upper_bound=config.architecture.lipschitz_constant_upper_bound,
        ).to(device=device)

        dynamics_model.compute_normalization_stats(
            state_batch=torch.from_numpy(train_transitions.states).to(device=device),
            action_batch=torch.from_numpy(train_transitions.actions).to(device=device),
            next_state_batch=torch.from_numpy(train_transitions.next_states).to(
                device=device
            ),
        )

        (
            train_state_windows,  # (window_count, window_length + 1, state_dimensionality)
            train_action_windows,  # (window_count, window_length, action_dimensionality)
        ) = build_windows(
            episodes=training_episodes, window_length=config.data.loader.window_length
        )
        (validation_state_windows, validation_action_windows) = build_windows(
            episodes=validation_episodes, window_length=config.data.loader.window_length
        )
        logger.info(
            "Built %d training windows of length %d.",
            train_action_windows.shape[0],
            config.data.loader.window_length,
        )

        best_rollout_error, best_epoch = _train_dynamics_model(
            dynamics_model=dynamics_model,
            training_state_windows=train_state_windows,
            training_action_windows=train_action_windows,
            validation_state_windows=validation_state_windows,
            validation_action_windows=validation_action_windows,
            training_states=train_transitions.states,
            training_actions=train_transitions.actions,
            validation_states=validation_transitions.states,
            validation_actions=validation_transitions.actions,
            config=config,
            run=run,
        )

        run.summary["selected/epoch"] = best_epoch
        run.summary["selected/rollout_error"] = best_rollout_error

        _log_model_artifact(dynamics_model=dynamics_model, config=config, run=run)


def _train_dynamics_model(
    dynamics_model: DynamicsModel,
    training_state_windows: npt.NDArray[np.float32],
    training_action_windows: npt.NDArray[np.float32],
    validation_state_windows: npt.NDArray[np.float32],
    validation_action_windows: npt.NDArray[np.float32],
    training_states: npt.NDArray[
        np.float32
    ],  # (training_sample_count, state_dimensionality)
    training_actions: npt.NDArray[
        np.float32
    ],  # (training_sample_count, action_dimensionality)
    validation_states: npt.NDArray[
        np.float32
    ],  # (validation_sample_count, state_dimensionality)
    validation_actions: npt.NDArray[
        np.float32
    ],  # (validation_sample_count, action_dimensionality)
    config: Config,
    run: Run,
) -> tuple[float, int]:
    model_device = next(dynamics_model.parameters()).device

    train_state_windows_tensor = torch.from_numpy(training_state_windows).to(
        device=model_device
    )
    train_action_windows_tensor = torch.from_numpy(training_action_windows).to(
        device=model_device
    )

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

    # AveragedModel.module is typed as nn.Module but it is the DynamicsModel we wrapped.
    ema_dynamics_model = cast(DynamicsModel, ema_model.module)

    window_count = train_state_windows_tensor.shape[0]
    steps_per_epoch = max(window_count // config.data.loader.batch_size, 1)

    global_step = 0

    best_validation_loss = float("inf")
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_epoch_index = -1

    for epoch_index in tqdm(
        range(config.training.epoch_count), desc="training", unit="epoch"
    ):
        dynamics_model.train()

        shuffled_window_indices = torch.randperm(n=window_count, device=model_device)

        for step_index in range(steps_per_epoch):
            batch_window_indices = shuffled_window_indices[
                step_index * config.data.loader.batch_size : (step_index + 1)
                * config.data.loader.batch_size
            ]

            optimizer.zero_grad()

            loss = compute_loss(
                dynamics_model=dynamics_model,
                state_window_batch=train_state_windows_tensor[batch_window_indices],
                action_window_batch=train_action_windows_tensor[batch_window_indices],
            )
            loss.backward()

            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters=dynamics_model.parameters(), max_norm=1.0
            )
            optimizer.step()

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
            validation_state_windows=validation_state_windows,
            validation_action_windows=validation_action_windows,
            batch_size=config.data.loader.batch_size,
        )

        run.log({"loss/validation": validation_loss}, step=global_step)

        run.log(
            {
                **_compute_lipschitz_metrics(
                    dynamics_model=ema_dynamics_model,
                    states=training_states,
                    actions=training_actions,
                    batch_size=config.data.loader.batch_size,
                    split="training",
                ),
                **_compute_lipschitz_metrics(
                    dynamics_model=ema_dynamics_model,
                    states=validation_states,
                    actions=validation_actions,
                    batch_size=config.data.loader.batch_size,
                    split="validation",
                ),
            },
            step=global_step,
        )

        run.log({"epoch": epoch_index}, step=global_step)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state_dict = copy.deepcopy(ema_dynamics_model.state_dict())
            best_epoch_index = epoch_index

    if best_state_dict is None:
        raise ValueError("No checkpoint was recorded")

    dynamics_model.load_state_dict(best_state_dict)
    logger.info(
        "Restored best EMA checkpoint from epoch %d (rollout error %.6f).",
        best_epoch_index,
        best_validation_loss,
    )
    return best_validation_loss, best_epoch_index


def _compute_lipschitz_metrics(
    dynamics_model: DynamicsModel,
    states: npt.NDArray[np.float32],  # (sample_count, state_dimensionality)
    actions: npt.NDArray[np.float32],  # (sample_count, action_dimensionality)
    batch_size: int,
    split: str,
) -> dict[str, float | wandb.Histogram]:
    local_lipschitz_constants = calculate_local_lipschitz_constants(
        dynamics_model=dynamics_model,
        states=states,
        actions=actions,
        batch_size=batch_size,
    )
    return {
        f"lipschitz/distribution_{split}": wandb.Histogram(
            local_lipschitz_constants.tolist()
        ),
        f"lipschitz/mean_{split}": float(np.mean(local_lipschitz_constants)),
        f"lipschitz/p99_{split}": float(
            np.percentile(local_lipschitz_constants, q=99.0)
        ),
    }


@torch.no_grad()
def _refresh_spectral_norm_estimates(ema_dynamics_model: DynamicsModel) -> None:
    """Recompute the spectral-norm vectors for the EMA model's averaged weights.

    These vectors only update during train-mode forward passes, and the EMA model is
    never trained, so its vectors are out of date. They also can't be copied from the
    live model, because they only fit the exact weights they were computed from, and
    the EMA model has different weights. So we run a few forward
    passes in train mode: each pass moves the vectors one step closer to the correct
    values for the current weights.
    The input values play no role here, so we use zeros.
    """
    model_device = next(ema_dynamics_model.parameters()).device

    dummy_state_batch = torch.zeros(
        1, ema_dynamics_model.state_dimensionality, device=model_device
    )
    dummy_action_batch = torch.zeros(
        1, ema_dynamics_model.action_dimensionality, device=model_device
    )

    model_was_training = ema_dynamics_model.training
    ema_dynamics_model.train()
    for _ in range(50):
        ema_dynamics_model.predict_next_state(
            state_batch=dummy_state_batch, action_batch=dummy_action_batch
        )
    ema_dynamics_model.train(mode=model_was_training)


def _log_model_artifact(
    dynamics_model: DynamicsModel, config: Config, run: Run
) -> None:
    config_hash = compute_config_hash(config=asdict(config))
    with tempfile.TemporaryDirectory() as temporary_directory:
        dynamics_checkpoint_path = Path(temporary_directory) / "dynamics.pt"
        save_dynamics_checkpoint(
            dynamics_model=dynamics_model, path=dynamics_checkpoint_path
        )

        model_artifact = wandb.Artifact(
            name=f"dynamics-model-{config_hash}", type="model"
        )
        model_artifact.add_file(
            local_path=str(dynamics_checkpoint_path), name="dynamics.pt"
        )
        run.log_artifact(model_artifact).wait()


if __name__ == "__main__":
    main()
