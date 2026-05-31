import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import tyro
import wandb
from numpy import typing as npt
from rich.pretty import pretty_repr
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from wandb import Run

from scripts.ashared import compute_config_hash, create_logger, set_seed
from scripts.data.augmentation.ccil.dynamics.checkpoint import save_checkpoint
from scripts.data.augmentation.ccil.dynamics.config import Config
from scripts.data.augmentation.ccil.dynamics.data import (
    build_windows,
    concatenate_episodes,
    load_episodes,
)
from scripts.data.augmentation.ccil.dynamics.evaluation import (
    calculate_local_lipschitz_constants,
    evaluate_rollout,
)
from scripts.data.augmentation.ccil.dynamics.losses import compute_loss
from scripts.data.augmentation.ccil.dynamics_model import DynamicsModel
from scripts.shared.early_stoppong import EarlyStopping
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH

logger = create_logger()


def main():
    config = tyro.cli(f=Config, config=[tyro.conf.UsePythonSyntaxForLiteralCollections])
    logger.info(pretty_repr(config))
    _run(config=config)


def _run(config: Config) -> None:
    set_seed(seed=config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using torch device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        logger.info(
            "GPU Memory: %.2f GB",
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )

    with wandb.init(
        project="abc",
        job_type="dynamics-train",
        config=asdict(config),
        dir=OUTPUT_DIRECTORY_PATH,
    ) as run:
        train_episodes, validation_episodes = train_test_split(
            load_episodes(
                artifact=run.use_artifact(artifact_or_name=config.dataset_artifact),
                logger=logger,
            ),
            test_size=config.evaluation.data_split.holdout_fraction,
            random_state=config.evaluation.data_split.seed,
            shuffle=True,
        )
        logger.info("# Train episodes: %d", len(train_episodes))
        logger.info("# Validation episodes: %d", len(validation_episodes))

        train_transitions = concatenate_episodes(episodes=train_episodes)
        (
            train_state_windows,  # (window_count, window_length + 1, state_dimensionality)
            train_action_windows,  # (window_count, window_length, action_dimensionality)
        ) = build_windows(
            episodes=train_episodes, window_length=config.training.loss.window_length
        )
        logger.info(
            "Built %d training windows of length %d.",
            train_action_windows.shape[0],
            config.training.loss.window_length,
        )

        dynamics_model = DynamicsModel(
            state_dimensionality=train_transitions.states.shape[1],
            action_dimensionality=train_transitions.actions.shape[1],
            hidden_dimensionalities=config.architecture.hidden_layer_sizes,
            activation_function_name=config.architecture.activation,
            lipschitz_constant_upper_bound=config.architecture.lipschitz_constant_upper_bound,
        ).to(device=device)

        dynamics_model.compute_normalization_stats(
            state_batch=torch.from_numpy(train_transitions.states).to(device=device),
            action_batch=torch.from_numpy(train_transitions.actions).to(device=device),
            next_state_batch=torch.from_numpy(train_transitions.next_states).to(
                device=device
            ),
        )

        best_backward_error, best_epoch = _train_dynamics_model(
            dynamics_model=dynamics_model,
            training_state_windows=train_state_windows,
            training_action_windows=train_action_windows,
            validation_episodes=validation_episodes,
            config=config,
            run=run,
        )

        run.summary["selected/epoch"] = best_epoch
        run.summary["selected/clean_backward_error"] = best_backward_error

        _log_model_artifact(dynamics_model=dynamics_model, config=config, run=run)


def _train_dynamics_model(
    dynamics_model: DynamicsModel,
    training_state_windows: npt.NDArray[np.float32],
    training_action_windows: npt.NDArray[np.float32],
    validation_episodes: list,
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

    window_count = train_state_windows_tensor.shape[0]
    steps_per_epoch = max(window_count // config.training.batch_size, 1)

    early_stopping = EarlyStopping(patience=config.training.early_stopping_patience)

    global_step = 0
    for epoch_index in tqdm(
        range(config.training.epoch_count), desc="training", unit="epoch"
    ):
        dynamics_model.train()

        shuffled_window_indices = torch.randperm(n=window_count, device=model_device)

        for step_index in range(steps_per_epoch):
            batch_window_indices = shuffled_window_indices[
                step_index * config.training.batch_size : (step_index + 1)
                * config.training.batch_size
            ]

            optimizer.zero_grad()

            total_loss, loss_breakdown = compute_loss(
                dynamics_model=dynamics_model,
                state_window_batch=train_state_windows_tensor[batch_window_indices],
                action_window_batch=train_action_windows_tensor[batch_window_indices],
                config=config.training.loss,
            )
            total_loss.backward()

            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters=dynamics_model.parameters(), max_norm=10.0
            )
            optimizer.step()

            run.log(
                {
                    "loss/total": loss_breakdown.total.item(),
                    "loss/forward": loss_breakdown.forward.item(),
                    "loss/inverse": loss_breakdown.inverse.item(),
                    "gradient_norm": gradient_norm.item(),
                },
                step=global_step,
            )

            global_step += 1

        reconstruction_error, convergence_rate, lipschitz_violation_rate = (
            _evaluate_dynamics_model(
                dynamics_model=dynamics_model,
                validation_episodes=validation_episodes,
                config=config,
            )
        )

        run.log(
            {
                "eval/reconstruction_error": reconstruction_error,
                "eval/convergence_rate": convergence_rate,
                "eval/lipschitz_violation_rate": lipschitz_violation_rate,
            },
            step=global_step,
        )

        should_stop = early_stopping.update(
            metric=reconstruction_error, model=dynamics_model, epoch=epoch_index
        )
        if should_stop:
            logger.info(
                "Early stopping at epoch %d (best epoch %d, backward prediction error %.6f).",
                epoch_index,
                early_stopping.best_epoch,
                early_stopping.best_metric,
            )
            break

    if early_stopping.best_state_dict is None:
        raise ValueError()

    dynamics_model.load_state_dict(early_stopping.best_state_dict)
    return early_stopping.best_metric, early_stopping.best_epoch


def _evaluate_dynamics_model(
    dynamics_model: DynamicsModel, validation_episodes: list, config: Config
) -> tuple[float, float, float]:

    dynamics_model.eval()

    rollout_result = evaluate_rollout(
        dynamics_model=dynamics_model,
        episodes=validation_episodes,
        window_length=config.evaluation.window_length,
        max_newton_iteration_count=config.evaluation.max_newton_iteration_count,
    )

    validation_transitions = concatenate_episodes(episodes=validation_episodes)
    local_lipschitz_constants = calculate_local_lipschitz_constants(
        dynamics_model=dynamics_model,
        states=validation_transitions.states,
        actions=validation_transitions.actions,
        batch_size=1_024,
    )

    return (
        rollout_result.reconstruction_error,
        rollout_result.newton_convergence_rate,
        float(
            np.mean(
                local_lipschitz_constants
                > config.architecture.lipschitz_constant_upper_bound
            )
        ),
    )


def _log_model_artifact(
    dynamics_model: DynamicsModel, config: Config, run: Run
) -> None:
    """Log the selected model to wandb only. Serialise into a temp dir, push the artifact,
    await the upload, discard the temp dir. Name is a config hash; no metadata (the run
    carries the config, and lineage comes from use_artifact/log_artifact).
    """
    with tempfile.TemporaryDirectory() as temporary_directory:
        save_checkpoint(
            dynamics_model=dynamics_model,
            output_directory_path=Path(temporary_directory),
            run=run,
        )

        artifact = wandb.Artifact(
            name=f"dynamics-model-{compute_config_hash(config=asdict(config))}",
            type="model",
        )
        artifact.add_dir(temporary_directory)
        logged_artifact = run.log_artifact(artifact)
        logged_artifact.wait()  # block until uploaded, so the temp dir is safe to delete


if __name__ == "__main__":
    main()
