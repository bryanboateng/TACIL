import numpy as np
import torch
import wandb
from numpy import typing as npt
from wandb import Run

from scripts.data.augmentation.tacil.create_data.distances import (
    calculate_median_state_step_distance,
)
from scripts.data.augmentation.tacil.create_dynamics.data import Transitions
from scripts.data.augmentation.tacil.create_dynamics.dynamics_model_training import (
    EpochMetrics,
)
from scripts.data.augmentation.tacil.create_dynamics.statistics import plots
from scripts.data.augmentation.tacil.create_dynamics.statistics.error_calculation import (
    calculate_errors,
)
from scripts.data.augmentation.tacil.dynamics.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)
from scripts.environments.tacil_dynamics_plotter import TacilDynamicsPlotter
from scripts.shared import stage_names
from scripts.shared.distribution_summary import summarize_distribution
from scripts.shared.episodes import Episodes
from scripts.shared.log import create_logger

logger = create_logger()

STAGE_NAME = stage_names.DYNAMICS

EPOCH_METRIC_NAME = f"{STAGE_NAME}/0_epoch"

LOSS_SECTION = f"{STAGE_NAME}/1_loss"
GRADIENT_NORM_SECTION = f"{STAGE_NAME}/2_gradient_norm"
LIPSCHITZ_SECTION = f"{STAGE_NAME}/7_lipschitz"

SUMMARY_STEP_COUNTS = (1, 2, 3, 5, 10, 20)


def define_epoch_metrics(*, run: Run) -> None:
    run.define_metric(name=EPOCH_METRIC_NAME)
    for section in (LOSS_SECTION, GRADIENT_NORM_SECTION, LIPSCHITZ_SECTION):
        run.define_metric(name=f"{section}/*", step_metric=EPOCH_METRIC_NAME)


def log_epoch(*, epoch_index: int, metrics: EpochMetrics, run: Run) -> None:
    logged_metrics: dict[str, float | wandb.Histogram] = {
        EPOCH_METRIC_NAME: epoch_index
    }
    for prefix, member_values in (
        (f"{LOSS_SECTION}/1_training", metrics.training_losses),
        (f"{LOSS_SECTION}/2_validation", metrics.validation_losses),
        (GRADIENT_NORM_SECTION, metrics.gradient_norms),
    ):
        for label, value in _summarize_finite_values(values=member_values).items():
            logged_metrics[f"{prefix}/{label}"] = value

    for split, local_lipschitz_constants in (
        ("1_training", metrics.training_local_lipschitz_constants),
        ("2_validation", metrics.validation_local_lipschitz_constants),
    ):
        finite_constants = local_lipschitz_constants[
            np.isfinite(local_lipschitz_constants)
        ]  # (finite_count,)
        if finite_constants.size > 0:
            logged_metrics[f"{LIPSCHITZ_SECTION}/{split}"] = wandb.Histogram(
                sequence=finite_constants.tolist()
            )

    run.log(data=logged_metrics)


def log_best_checkpoint_summary(
    *,
    best_validation_losses: npt.NDArray[np.float64],  # (member_count,)
    best_epochs: npt.NDArray[np.int64],  # (member_count,)
    run: Run,
) -> None:
    converged_member_mask = np.isfinite(best_validation_losses)  # (member_count,)

    for name, values in (
        ("1_validation_loss", best_validation_losses[converged_member_mask]),
        ("2_epoch", best_epochs[converged_member_mask]),
    ):
        for label, value in _summarize_finite_values(values=values).items():
            run.summary[f"{STAGE_NAME}/3_best_checkpoint/{name}/{label}"] = value
    run.summary[f"{STAGE_NAME}/3_best_checkpoint/3_diverged_member_count"] = int(
        np.count_nonzero(~converged_member_mask)
    )


def log_window_counts(
    *, training_window_count: int, validation_window_count: int, run: Run
) -> None:
    run.summary[f"{STAGE_NAME}/4_window_count/1_training"] = training_window_count
    run.summary[f"{STAGE_NAME}/4_window_count/2_validation"] = validation_window_count


def log_ensemble_statistics(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    dynamics_plotter: TacilDynamicsPlotter,
    validation_transitions_per_episode: list[Transitions],
    episodes: Episodes,
    run: Run,
) -> None:
    median_expert_state_step_distance = calculate_median_state_step_distance(
        episodes=episodes, dynamics_model_ensemble=dynamics_model_ensemble
    )
    error_measurements = calculate_errors(
        dynamics_model_ensemble=dynamics_model_ensemble,
        validation_transitions_per_episode=validation_transitions_per_episode,
        convergence_tolerance=1e-3 * median_expert_state_step_distance,
    )

    backward_replay_errors_per_step_count = (
        error_measurements.backward_replay_errors.cpu().numpy()
    )  # (window_count, step_count)
    backward_account_errors_per_step_count = (
        error_measurements.backward_account_errors.cpu().numpy()
    )  # (window_count, step_count)
    for errors_per_step_count, key_suffix in (
        (backward_replay_errors_per_step_count, "1_replay"),
        (backward_account_errors_per_step_count, "2_account"),
    ):
        _log_errors_per_step_count(
            errors_per_step_count=errors_per_step_count,
            key_prefix=f"{STAGE_NAME}/6_stopping_rule_error/{key_suffix}",
            run=run,
        )

    run.summary[
        f"{STAGE_NAME}/5_error_over_step_count/1_backward/0_complete_walk_fraction"
    ] = float(
        np.isfinite(
            error_measurements.backward_prediction_errors[:, -1].cpu().numpy()
        ).mean()
    )

    for direction_name, prediction_errors in (
        ("1_backward", error_measurements.backward_prediction_errors),
        ("2_forward", error_measurements.forward_prediction_errors),
    ):
        _log_errors_per_step_count(
            errors_per_step_count=prediction_errors.cpu().numpy(),
            key_prefix=f"{STAGE_NAME}/5_error_over_step_count/{direction_name}",
            run=run,
        )
    run.log(
        data={
            f"{STAGE_NAME}/5_error_over_step_count/0_plot": plots.create_prediction_error_vs_step_count_plot(
                forward_prediction_errors_per_step_count=(
                    error_measurements.forward_prediction_errors.cpu().numpy()
                ),
                backward_prediction_errors_per_step_count=(
                    error_measurements.backward_prediction_errors.cpu().numpy()
                ),
            ),
            f"{STAGE_NAME}/6_stopping_rule_error/0_plot": (
                plots.create_stopping_rule_error_plot(
                    backward_prediction_errors_per_step_count=(
                        error_measurements.backward_prediction_errors.cpu().numpy()
                    ),
                    backward_replay_errors_per_step_count=(
                        backward_replay_errors_per_step_count
                    ),
                    backward_account_errors_per_step_count=(
                        backward_account_errors_per_step_count
                    ),
                )
            ),
            **_create_replay_panels(
                dynamics_model_ensemble=dynamics_model_ensemble,
                dynamics_plotter=dynamics_plotter,
                validation_transitions_per_episode=validation_transitions_per_episode,
            ),
        }
    )


# ---- Private --------------------------------------------


def _log_errors_per_step_count(
    *,
    errors_per_step_count: npt.NDArray[np.float32],  # (window_count, step_count)
    key_prefix: str,
    run: Run,
) -> None:
    for step_count in SUMMARY_STEP_COUNTS:
        if step_count > errors_per_step_count.shape[1]:
            continue

        # A walk that stopped earlier than this has no error to contribute.
        errors = errors_per_step_count[:, step_count - 1]
        errors = errors[np.isfinite(errors)]
        if errors.size == 0:
            continue

        for label, value in summarize_distribution(values=errors).items():
            run.summary[f"{key_prefix}/{step_count:02d}/{label}"] = value


def _create_replay_panels(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    dynamics_plotter: TacilDynamicsPlotter,
    validation_transitions_per_episode: list[Transitions],
) -> dict[str, wandb.Image]:
    device = next(dynamics_model_ensemble.parameters()).device

    panel_count = 4

    replay_panels: dict[str, wandb.Image] = {}
    # A small dataset can leave fewer validation episodes than there are panels.
    for episode_index in range(
        min(panel_count, len(validation_transitions_per_episode))
    ):
        episode = validation_transitions_per_episode[episode_index]

        true_trajectory = np.concatenate(
            [episode.states, episode.next_states[[-1]]], axis=0
        )  # (step_count + 1, state_dimensionality)
        actions = torch.from_numpy(episode.actions).to(device=device)

        forward_member_trajectories = _roll_out_members_forward(
            dynamics_model_ensemble=dynamics_model_ensemble,
            initial_state=torch.from_numpy(true_trajectory[0]).to(device=device),
            actions=actions,
        )

        episode_plots = dynamics_plotter(
            member_trajectories=forward_member_trajectories.cpu().numpy(),
            true_trajectory=true_trajectory,
            episode_index=episode_index,
        )
        for name, image in episode_plots.items():
            replay_panels[f"{STAGE_NAME}/8_replay/episode_{episode_index}/{name}"] = (
                image
            )
    return replay_panels


def _roll_out_members_forward(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    initial_state: torch.Tensor,  # (state_dimensionality,)
    actions: torch.Tensor,  # (step_count, action_dimensionality)
) -> torch.Tensor:  # (member_count, step_count + 1, state_dimensionality)
    per_member_state_batch = initial_state.expand(
        dynamics_model_ensemble.member_count, 1, -1
    )  # (member_count, 1, state_dimensionality)
    trajectory_states = [per_member_state_batch]
    for step_index in range(actions.shape[0]):
        per_member_state_batch = dynamics_model_ensemble.predict_next_state_per_member(
            state_batch=per_member_state_batch,
            action_batch=actions[step_index].unsqueeze(dim=0),
        )  # (member_count, 1, state_dimensionality)
        trajectory_states.append(per_member_state_batch)

    return torch.cat(tensors=trajectory_states, dim=1)


def _summarize_finite_values(
    *, values: npt.NDArray[np.floating] | npt.NDArray[np.integer]
) -> dict[str, float]:
    finite_values = values[np.isfinite(values)]  # (finite_count,)
    if finite_values.size == 0:
        return {}
    return summarize_distribution(values=finite_values)
