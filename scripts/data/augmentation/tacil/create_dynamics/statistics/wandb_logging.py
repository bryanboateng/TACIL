import numpy as np
import torch
import wandb
from wandb import Run

from scripts.data.augmentation.tacil.create_data.step_distance import (
    calculate_median_step_distance,
)
from scripts.data.augmentation.tacil.create_dynamics.data import TransitionSet
from scripts.data.augmentation.tacil.create_dynamics.dynamics_model_training import (
    MemberEpochMetrics,
    MemberTrainingState,
)
from scripts.data.augmentation.tacil.create_dynamics.statistics import plots
from scripts.data.augmentation.tacil.create_dynamics.statistics.chain_measurement import (  # noqa: E501
    draw_chain_windows,
    measure_backward_chains,
    measure_forward_chains,
    roll_out_members_forward,
)
from scripts.data.augmentation.tacil.dynamics.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)
from scripts.shared.distribution_summary import summarize_distribution
from scripts.shared.episode_store import EpisodeStore
from scripts.shared.log import create_logger

logger = create_logger()

STAGE_NAME = "1_dynamics"

EPOCH_METRIC_NAME = f"{STAGE_NAME}/0_epoch"

_REPLAY_EPISODE_COUNT = 4
_REPLAY_STEP_COUNT = 100

# Where the error curve is sampled for the summary. One step is the model
# without any compounding, twenty is the deepest measured, and the ones in
# between catch a curve that only bends late. The early ones sit close
# together because that is where compounding starts and the curve is steepest.
_SUMMARY_STEP_COUNTS = (1, 2, 3, 5, 10, 20)


def define_epoch_metrics(*, run: Run) -> None:
    run.define_metric(name=EPOCH_METRIC_NAME)
    for metric_pattern in ("1_loss/*", "2_gradient_norm/*", "7_lipschitz/*"):
        run.define_metric(
            name=f"{STAGE_NAME}/{metric_pattern}", step_metric=EPOCH_METRIC_NAME
        )


def log_window_counts(
    *, member_training_window_counts: list[int], validation_window_count: int, run: Run
) -> None:
    for label, value in summarize_distribution(
        values=np.asarray(member_training_window_counts)
    ).items():
        run.summary[f"{STAGE_NAME}/4_window_count/training/{label}"] = value
    run.summary[f"{STAGE_NAME}/4_window_count/validation"] = validation_window_count


def log_epoch(
    *, epoch_index: int, per_member_metrics: list[MemberEpochMetrics], run: Run
) -> None:
    logged_metrics: dict[str, float | wandb.Histogram] = {
        EPOCH_METRIC_NAME: epoch_index
    }
    # Summarized over the members rather than one curve each: twenty curves per
    # quantity are unreadable, and what the curves are watched for — does it
    # come down, do the members stay together — the summary says too.
    for name, member_values in (
        ("1_loss/training", [m.training_loss for m in per_member_metrics]),
        ("1_loss/validation", [m.validation_loss for m in per_member_metrics]),
        ("2_gradient_norm", [m.gradient_norm for m in per_member_metrics]),
    ):
        summary = summarize_distribution(
            values=np.asarray(member_values, dtype=np.float64)
        )
        for label, value in summary.items():
            logged_metrics[f"{STAGE_NAME}/{name}/{label}"] = value

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
        logged_metrics[f"{STAGE_NAME}/7_lipschitz/{split}"] = wandb.Histogram(
            sequence=np.concatenate(member_split_lipschitz).tolist()
        )

    run.log(data=logged_metrics)


def log_checkpoint_summary(
    *, member_states: list[MemberTrainingState], run: Run
) -> None:
    checkpoints = []
    for member_state in member_states:
        if member_state.best_checkpoint is None:
            raise ValueError("No checkpoint was recorded")
        checkpoints.append(member_state.best_checkpoint)

    validation_losses = np.array(
        [checkpoint.validation_loss for checkpoint in checkpoints]
    )
    epochs = np.array([checkpoint.epoch for checkpoint in checkpoints])

    # Each member keeps the epoch with its lowest validation loss, and this is
    # what those kept epochs look like across the members. The spread shows how
    # much the result depends on the initialization of a single one.
    for name, values in (("validation_loss", validation_losses), ("epoch", epochs)):
        for label, value in summarize_distribution(values=values).items():
            run.summary[f"{STAGE_NAME}/3_best_checkpoint/{name}/{label}"] = value


def log_ensemble_statistics(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    validation_episodes: list[TransitionSet],
    episode_store: EpisodeStore,
    run: Run,
) -> None:
    device = next(dynamics_model_ensemble.parameters()).device

    median_expert_step_distance = calculate_median_step_distance(
        episode_store=episode_store, dynamics_model_ensemble=dynamics_model_ensemble
    )

    state_windows, action_windows = draw_chain_windows(
        validation_episodes=validation_episodes
    )
    state_window_batch = torch.from_numpy(state_windows).to(device=device)
    action_window_batch = torch.from_numpy(action_windows).to(device=device)

    logger.info("Diagnosing on %d held-out chains.", state_window_batch.shape[0])
    forward_chain_measurements = measure_forward_chains(
        dynamics_model_ensemble=dynamics_model_ensemble,
        state_windows=state_window_batch,
        action_windows=action_window_batch,
    )
    backward_chain_measurements = measure_backward_chains(
        dynamics_model_ensemble=dynamics_model_ensemble,
        state_windows=state_window_batch,
        action_windows=action_window_batch,
        convergence_tolerance=1e-3 * median_expert_step_distance,
    )

    # What the limit would be read against after each number of steps, sampled
    # at the same counts as the error it is meant to catch, so the two can be
    # held against each other. In normalized state units, like the error,
    # because that comparison is the whole point.
    #
    # It also carries the run-is-worthless alarm: members that collapsed onto
    # one function all replay onto the anchor, so the signal falls to zero and
    # every limit passes. Measured in the sweep of 2026-08-16, a lower
    # Lipschitz bound lengthened the chains even at a stricter limit
    # (r = -0.39, p = 0.004 with the limit partialled out; bound under 10 drew
    # a median limit of 11.6 and still reached length 9.5, against 17.7 and
    # 8.0 above it). Those runs scored 0.59 against 0.69.
    backward_replay_errors_per_step_count = (
        backward_chain_measurements.replay_errors.cpu().numpy()
    )  # (chain_count, step_count)
    for step_count in _SUMMARY_STEP_COUNTS:
        if step_count > backward_replay_errors_per_step_count.shape[1]:
            continue
        replay_errors_at_step_count = backward_replay_errors_per_step_count[
            :, step_count - 1
        ]
        # A chain that stopped earlier than this has nothing to contribute.
        replay_errors_at_step_count = replay_errors_at_step_count[
            np.isfinite(replay_errors_at_step_count)
        ]
        if replay_errors_at_step_count.size == 0:
            continue
        for label, value in summarize_distribution(
            values=replay_errors_at_step_count
        ).items():
            run.summary[f"{STAGE_NAME}/6_replay_error/{step_count:02d}/{label}"] = value

    # How much of the long backward numbers to believe. A chain dies at its
    # first non-converged solve, so anything below 1 means the errors after
    # more steps were measured on the subset that survived — the easier one.
    run.summary[
        f"{STAGE_NAME}/5_error_over_step_count/backward/complete_chain_fraction"
    ] = float(
        np.isfinite(
            backward_chain_measurements.prediction_errors[:, -1].cpu().numpy()
        ).mean()
    )

    # A few points off the error curve, so runs can be ranked on a number
    # instead of on a picture.
    for direction_name, measurements in (
        ("backward", backward_chain_measurements),
        ("forward", forward_chain_measurements),
    ):
        for step_count in _SUMMARY_STEP_COUNTS:
            if step_count > measurements.prediction_errors.shape[1]:
                continue
            errors_at_step_count = (
                measurements.prediction_errors[:, step_count - 1].cpu().numpy()
            )
            # A chain that stopped earlier than this has no error to contribute.
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
                    forward_chain_measurements.prediction_errors.cpu().numpy()
                ),
                backward_prediction_errors_per_step_count=(
                    backward_chain_measurements.prediction_errors.cpu().numpy()
                ),
            ),
            f"{STAGE_NAME}/6_replay_error/plot": (
                plots.create_replay_error_plot(
                    backward_prediction_errors_per_step_count=(
                        backward_chain_measurements.prediction_errors.cpu().numpy()
                    ),
                    backward_replay_errors_per_step_count=(
                        backward_replay_errors_per_step_count
                    ),
                )
            ),
            **_replay_panels(
                dynamics_model_ensemble=dynamics_model_ensemble,
                validation_episodes=validation_episodes,
                convergence_tolerance=1e-3 * median_expert_step_distance,
                device=device,
            ),
        }
    )


# ---- Private --------------------------------------------


def _replay_panels(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    validation_episodes: list[TransitionSet],
    convergence_tolerance: float,
    device: torch.device,
) -> dict[str, wandb.Image]:
    replay_panels: dict[str, wandb.Image] = {}
    for episode_index in range(min(_REPLAY_EPISODE_COUNT, len(validation_episodes))):
        episode = validation_episodes[episode_index]
        step_count = min(_REPLAY_STEP_COUNT, episode.actions.shape[0])

        true_trajectory = np.concatenate(
            [episode.states[:step_count], episode.next_states[[step_count - 1]]], axis=0
        )  # (step_count + 1, state_dimensionality)
        actions = torch.from_numpy(episode.actions[:step_count]).to(device=device)

        forward_member_trajectories = roll_out_members_forward(
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
