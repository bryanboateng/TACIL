from dataclasses import asdict

import numpy as np
import torch
import tyro
import wandb
from numpy import typing as npt
from rich.pretty import pretty_repr
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from wandb import Run

from scripts.ashared import create_logger, set_seed
from scripts.data.augmentation.ccil.dynamics.checkpoint import save_checkpoint
from scripts.data.augmentation.ccil.dynamics.config import (
    ExperimentConfig,
    LossConfig,
    TrainingConfig,
)
from scripts.data.augmentation.ccil.dynamics.data import (
    build_windows,
    concatenate_episodes,
    load_episodes,
)
from scripts.data.augmentation.ccil.dynamics.evaluation import (
    calculate_local_lipschitz_constants,
    evaluate_backward_rollout,
    evaluate_forward_rollout,
)
from scripts.data.augmentation.ccil.dynamics.losses import LossBreakdown, compute_loss
from scripts.data.augmentation.ccil.dynamics.wandb_logging import (
    log_epoch,
    log_local_lipschitz_constant_distribution_plot,
    log_local_lipschitz_p99_and_violation_rate,
    log_rollout_error_plot,
    log_rollout_summaries,
)
from scripts.data.augmentation.ccil.dynamics_model import DynamicsModel
from scripts.data.augmentation.ccil.shared import CCIL_ROOT
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH

logger = create_logger()


def main():
    config = tyro.cli(
        f=ExperimentConfig, config=[tyro.conf.UsePythonSyntaxForLiteralCollections]
    )
    logger.info(pretty_repr(config))
    _run(config=config)


def _run(config: ExperimentConfig) -> None:
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
        job_type="create-dynamics-model",
        config=asdict(config),
        dir=OUTPUT_DIRECTORY_PATH,
    ) as run:
        assert run.name is not None

        train_episodes, validation_episodes = train_test_split(
            load_episodes(),
            test_size=config.evaluation.data_split.holdout_fraction,
            random_state=config.evaluation.data_split.seed,
            shuffle=True,
        )

        logger.info("# Train episodes: %d", len(train_episodes))
        logger.info("# Validation episodes: %d", len(validation_episodes))

        (
            train_states,  # (train_timestep_count, state_dimensionality)
            train_actions,  # (train_timestep_count, action_dimensionality)
            train_next_states,  # (train_timestep_count, state_dimensionality)
        ) = concatenate_episodes(episodes=train_episodes)

        window_length = config.loss.window_length

        (
            train_state_windows,  # (train_window_count, window_length + 1, state_dimensionality)
            train_action_windows,  # (train_window_count, window_length, action_dimensionality)
        ) = build_windows(episodes=train_episodes, window_length=window_length)

        logger.info(
            "Built %d training windows of length %d.",
            train_action_windows.shape[0],
            window_length,
        )

        dynamics_model = DynamicsModel(
            state_dimensionality=train_states.shape[1],
            action_dimensionality=train_actions.shape[1],
            hidden_dimensionalities=config.architecture.hidden_layer_sizes,
            activation_function_name=config.architecture.activation,
            lipschitz_constant_upper_bound=config.architecture.lipschitz_constant_upper_bound,
        ).to(device=device)

        dynamics_model.compute_normalization_stats(
            state_batch=torch.from_numpy(train_states).to(device=device),
            action_batch=torch.from_numpy(train_actions).to(device=device),
            next_state_batch=torch.from_numpy(train_next_states).to(device=device),
        )

        _train_dynamics_model(
            dynamics_model=dynamics_model,
            state_windows=train_state_windows,
            action_windows=train_action_windows,
            training_config=config.training,
            loss_config=config.loss,
            run=run,
        )

        save_checkpoint(
            dynamics_model=dynamics_model,
            output_directory_path=output_directory_path,
            run=run,
        )

        evaluation = config.evaluation

        # eval mode freezes the spectral_norm power iteration so repeated forward
        # passes during evaluation don't drift the weights away from the saved
        # checkpoint. Safe to leave set: evaluation is the last thing this function
        # does, so nothing trains afterward.
        dynamics_model.eval()

        forward_results = evaluate_forward_rollout(
            dynamics_model=dynamics_model,
            episodes=validation_episodes,
            horizon_lengths=evaluation.rollout_horizons,
            logger=logger,
        )
        clean_backward_rollout_results = evaluate_backward_rollout(
            dynamics_model=dynamics_model,
            episodes=validation_episodes,
            horizon_lengths=evaluation.rollout_horizons,
            max_newton_iteration_count=evaluation.backward_solver.max_newton_iteration_count,
            action_noise_scale=0.0,
            description="backward rollout (clean)",
        )
        noisy_backward_rollout_results = evaluate_backward_rollout(
            dynamics_model=dynamics_model,
            episodes=validation_episodes,
            horizon_lengths=evaluation.rollout_horizons,
            max_newton_iteration_count=evaluation.backward_solver.max_newton_iteration_count,
            action_noise_scale=evaluation.backward_solver.action_noise_scale,
            description="backward rollout (noisy)",
        )

        log_rollout_summaries(
            forward_rollout_errors=forward_results,
            clean_backward_rollout_results=clean_backward_rollout_results,
            noisy_backward_rollout_results=noisy_backward_rollout_results,
            objective_horizon_length=max(evaluation.rollout_horizons),
            run=run,
        )
        log_rollout_error_plot(
            forward_results=forward_results,
            clean_backward_rollout_results=clean_backward_rollout_results,
            noisy_backward_rollout_results=noisy_backward_rollout_results,
            run=run,
        )

        (
            validation_states,  # (validation_timestep_count, state_dimensionality)
            validation_actions,  # (validation_timestep_count, action_dimensionality)
            _,
        ) = concatenate_episodes(episodes=validation_episodes)

        local_lipschitz_constants = calculate_local_lipschitz_constants(
            dynamics_model=dynamics_model,
            states=validation_states,
            actions=validation_actions,
            batch_size=1_024,
        )

        log_local_lipschitz_constant_distribution_plot(
            local_lipschitz_constants=local_lipschitz_constants,
            lipschitz_constant_upper_bound=config.architecture.lipschitz_constant_upper_bound,
            run=run,
        )
        log_local_lipschitz_p99_and_violation_rate(
            local_lipschitz_constants=local_lipschitz_constants,
            lipschitz_constant_upper_bound=config.architecture.lipschitz_constant_upper_bound,
            run=run,
        )


def _train_dynamics_model(
    dynamics_model: DynamicsModel,
    state_windows: npt.NDArray[
        np.float32
    ],  # (window_count, window_length + 1, state_dimensionality)
    action_windows: npt.NDArray[
        np.float32
    ],  # (window_count, window_length , action_dimensionality)
    training_config: TrainingConfig,
    loss_config: LossConfig,
    run: Run,
) -> None:
    model_device = next(dynamics_model.parameters()).device

    state_windows_tensor = torch.from_numpy(state_windows).to(device=model_device)
    action_windows_tensor = torch.from_numpy(action_windows).to(device=model_device)

    parameters = list(dynamics_model.parameters())
    optimizer = torch.optim.Adam(
        params=parameters,
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    window_count = state_windows_tensor.shape[0]
    steps_per_epoch = max(window_count // training_config.batch_size, 1)

    for epoch_index in tqdm(
        range(training_config.epoch_count), desc="training", unit="epoch"
    ):
        shuffled_window_indices = torch.randperm(n=window_count, device=model_device)
        epoch_loss_breakdowns: list[LossBreakdown] = []
        epoch_gradient_norms: list[torch.Tensor] = []

        for step_index in range(steps_per_epoch):
            batch_window_indices = shuffled_window_indices[
                step_index * training_config.batch_size : (step_index + 1)
                * training_config.batch_size
            ]
            optimizer.zero_grad()

            total_loss, loss_breakdown = compute_loss(
                dynamics_model=dynamics_model,
                state_window_batch=state_windows_tensor[batch_window_indices],
                action_window_batch=action_windows_tensor[batch_window_indices],
                config=loss_config,
            )
            total_loss.backward()

            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters=parameters, max_norm=10.0
            )
            optimizer.step()

            epoch_loss_breakdowns.append(loss_breakdown)
            epoch_gradient_norms.append(gradient_norm.detach())

        log_epoch(
            loss_breakdowns=epoch_loss_breakdowns,
            gradient_norms=epoch_gradient_norms,
            epoch_index=epoch_index,
            run=run,
        )


if __name__ == "__main__":
    main()
