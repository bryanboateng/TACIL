import numpy as np
import torch
import wandb
from wandb import Run

from scripts.data.augmentation.tacil.create_data.step_distance import (
    calculate_median_step_distance,
)
from scripts.data.augmentation.tacil.create_dynamics.data import TransitionTriples
from scripts.data.augmentation.tacil.create_dynamics.dynamics_model_training import (
    MemberEpochMetrics,
    MemberTrainingState,
)
from scripts.data.augmentation.tacil.create_dynamics.statistics import plots
from scripts.data.augmentation.tacil.create_dynamics.statistics.error_measurement import (
    measure_errors,
)
from scripts.data.augmentation.tacil.dynamics.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)
from scripts.shared import stage_names
from scripts.shared.distribution_summary import summarize_distribution
from scripts.shared.episode_store import EpisodeStore
from scripts.shared.log import create_logger

logger = create_logger()

STAGE_NAME = stage_names.DYNAMICS

EPOCH_METRIC_NAME = f"{STAGE_NAME}/0_epoch"

LOSS_SECTION = f"{STAGE_NAME}/1_loss"
GRADIENT_NORM_SECTION = f"{STAGE_NAME}/2_gradient_norm"
LIPSCHITZ_SECTION = f"{STAGE_NAME}/7_lipschitz"


def define_epoch_metrics(*, run: Run) -> None:
    run.define_metric(name=EPOCH_METRIC_NAME)
    for section in (LOSS_SECTION, GRADIENT_NORM_SECTION, LIPSCHITZ_SECTION):
        run.define_metric(name=f"{section}/*", step_metric=EPOCH_METRIC_NAME)


def log_epoch(
    *, epoch_index: int, per_member_metrics: list[MemberEpochMetrics], run: Run
) -> None:
    logged_metrics: dict[str, float | wandb.Histogram] = {
        EPOCH_METRIC_NAME: epoch_index
    }
    for prefix, member_values in (
        (
            f"{LOSS_SECTION}/training",
            [metrics.training_loss for metrics in per_member_metrics],
        ),
        (
            f"{LOSS_SECTION}/validation",
            [metrics.validation_loss for metrics in per_member_metrics],
        ),
        (
            GRADIENT_NORM_SECTION,
            [metrics.gradient_norm for metrics in per_member_metrics],
        ),
    ):
        for label, value in summarize_distribution(
            values=np.asarray(member_values, dtype=np.float64)
        ).items():
            logged_metrics[f"{prefix}/{label}"] = value

    for split, member_split_lipschitz in (
        (
            "training",
            [
                metrics.training_local_lipschitz_constants
                for metrics in per_member_metrics
            ],
        ),
        (
            "validation",
            [
                metrics.validation_local_lipschitz_constants
                for metrics in per_member_metrics
            ],
        ),
    ):
        logged_metrics[f"{LIPSCHITZ_SECTION}/{split}"] = wandb.Histogram(
            sequence=np.concatenate(member_split_lipschitz).tolist()
        )

    run.log(data=logged_metrics)


def log_best_checkpoint_summary(
    *, member_states: list[MemberTrainingState], run: Run
) -> None:
    best_checkpoints = []
    for member_state in member_states:
        if member_state.best_checkpoint is None:
            raise ValueError("No checkpoint was recorded")
        best_checkpoints.append(member_state.best_checkpoint)

    for name, values in (
        (
            "validation_loss",
            np.array([checkpoint.validation_loss for checkpoint in best_checkpoints]),
        ),
        ("epoch", np.array([checkpoint.epoch for checkpoint in best_checkpoints])),
    ):
        for label, value in summarize_distribution(values=values).items():
            run.summary[f"{STAGE_NAME}/3_best_checkpoint/{name}/{label}"] = value


def log_window_counts(
    *, member_training_window_counts: list[int], validation_window_count: int, run: Run
) -> None:
    for label, value in summarize_distribution(
        values=np.asarray(member_training_window_counts)
    ).items():
        run.summary[f"{STAGE_NAME}/4_window_count/training/{label}"] = value
    run.summary[f"{STAGE_NAME}/4_window_count/validation"] = validation_window_count


def log_ensemble_statistics(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    validation_episodes: list[TransitionTriples],
    episode_store: EpisodeStore,
    run: Run,
) -> None:
    summary_step_counts = (1, 2, 3, 5, 10, 20)

    median_expert_step_distance = calculate_median_step_distance(
        episode_store=episode_store, dynamics_model_ensemble=dynamics_model_ensemble
    )
    error_measurements = measure_errors(
        dynamics_model_ensemble=dynamics_model_ensemble,
        validation_episodes=validation_episodes,
        convergence_tolerance=1e-3 * median_expert_step_distance,
    )

    backward_replay_errors_per_step_count = (
        error_measurements.backward_replay_errors.cpu().numpy()
    )  # (window_count, step_count)
    for step_count in summary_step_counts:
        if step_count > backward_replay_errors_per_step_count.shape[1]:
            continue
        replay_errors_at_step_count = backward_replay_errors_per_step_count[
            :, step_count - 1
        ]
        # A walk that stopped earlier than this has nothing to contribute.
        replay_errors_at_step_count = replay_errors_at_step_count[
            np.isfinite(replay_errors_at_step_count)
        ]
        if replay_errors_at_step_count.size == 0:
            continue
        for label, value in summarize_distribution(
            values=replay_errors_at_step_count
        ).items():
            run.summary[f"{STAGE_NAME}/6_replay_error/{step_count:02d}/{label}"] = value

    run.summary[
        f"{STAGE_NAME}/5_error_over_step_count/backward/complete_walk_fraction"
    ] = float(
        np.isfinite(
            error_measurements.backward_prediction_errors[:, -1].cpu().numpy()
        ).mean()
    )

    for direction_name, prediction_errors in (
        ("backward", error_measurements.backward_prediction_errors),
        ("forward", error_measurements.forward_prediction_errors),
    ):
        for step_count in summary_step_counts:
            if step_count > prediction_errors.shape[1]:
                continue
            errors_at_step_count = prediction_errors[:, step_count - 1].cpu().numpy()
            # A walk that stopped earlier than this has no error to contribute.
            errors_at_step_count = errors_at_step_count[
                np.isfinite(errors_at_step_count)
            ]
            if errors_at_step_count.size == 0:
                continue
            for label, value in summarize_distribution(
                values=errors_at_step_count
            ).items():
                run.summary[
                    f"{STAGE_NAME}/5_error_over_step_count/"
                    f"{direction_name}/{step_count:02d}/{label}"
                ] = value
    run.log(
        data={
            f"{STAGE_NAME}/5_error_over_step_count/plot": plots.create_prediction_error_vs_step_count_plot(
                forward_prediction_errors_per_step_count=(
                    error_measurements.forward_prediction_errors.cpu().numpy()
                ),
                backward_prediction_errors_per_step_count=(
                    error_measurements.backward_prediction_errors.cpu().numpy()
                ),
            ),
            f"{STAGE_NAME}/6_replay_error/plot": (
                plots.create_replay_error_plot(
                    backward_prediction_errors_per_step_count=(
                        error_measurements.backward_prediction_errors.cpu().numpy()
                    ),
                    backward_replay_errors_per_step_count=(
                        backward_replay_errors_per_step_count
                    ),
                )
            ),
            **_create_replay_panels(
                dynamics_model_ensemble=dynamics_model_ensemble,
                validation_episodes=validation_episodes,
            ),
        }
    )


# ---- Private --------------------------------------------


def _create_replay_panels(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    validation_episodes: list[TransitionTriples],
) -> dict[str, wandb.Image]:
    device = next(dynamics_model_ensemble.parameters()).device

    panel_count = 4

    replay_panels: dict[str, wandb.Image] = {}
    # A small dataset can leave fewer validation episodes than there are panels.
    for episode_index in range(min(panel_count, len(validation_episodes))):
        episode = validation_episodes[episode_index]

        true_trajectory = np.concatenate(
            [episode.states, episode.next_states[[-1]]], axis=0
        )  # (step_count + 1, state_dimensionality)
        actions = torch.from_numpy(episode.actions).to(device=device)

        forward_member_trajectories = _roll_out_members_forward(
            dynamics_model_ensemble=dynamics_model_ensemble,
            initial_state=torch.from_numpy(true_trajectory[0]).to(device=device),
            actions=actions,
        )

        replay_panels[f"{STAGE_NAME}/8_replay/episode_{episode_index}"] = (
            plots.create_true_dynamics_vs_model_plot(
                forward_member_trajectories=(forward_member_trajectories.cpu().numpy()),
                true_trajectory=true_trajectory,
                episode_index=episode_index,
            )
        )
    return replay_panels


@torch.no_grad()
def _roll_out_members_forward(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    initial_state: torch.Tensor,  # (state_dimensionality,)
    actions: torch.Tensor,  # (step_count, action_dimensionality)
) -> torch.Tensor:  # (member_count, step_count + 1, state_dimensionality)
    member_trajectories: list[torch.Tensor] = []
    for dynamics_model in dynamics_model_ensemble.dynamics_models:
        current_state_batch = initial_state.unsqueeze(dim=0)
        trajectory_states = [current_state_batch]
        for step_index in range(actions.shape[0]):
            current_state_batch = dynamics_model.predict_next_state(
                state_batch=current_state_batch,
                action_batch=actions[step_index].unsqueeze(dim=0),
            )
            trajectory_states.append(current_state_batch)
        member_trajectories.append(torch.cat(tensors=trajectory_states, dim=0))

    return torch.stack(tensors=member_trajectories, dim=0)
