from dataclasses import dataclass

import matplotlib
import numpy as np
import scipy.stats
import torch
import wandb
from numpy import typing as npt
from tqdm import tqdm
from wandb import Run

matplotlib.use(backend="Agg")
from matplotlib import patches  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

plt.style.use(style="seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300

from scripts.data.augmentation.tacil.create_data.inverse_dynamics import (  # noqa: E402
    solve_inverse_dynamics,
)
from scripts.data.augmentation.tacil.create_data.step_distance import (  # noqa: E402
    calculate_median_step_distance,
)
from scripts.data.augmentation.tacil.create_dynamics.data import (  # noqa: E402
    TransitionSet,
    build_windows,
)
from scripts.data.augmentation.tacil.models.dynamics_model_ensemble import (  # noqa: E402
    DynamicsModelEnsemble,
)
from scripts.shared.data import EpisodeStore  # noqa: E402
from scripts.shared.log import create_logger  # noqa: E402

logger = create_logger()

# Fixed measurement shape: deep enough to show the growth trend, small enough
# to stay affordable, and heavily overlapping windows make more chains
# redundant anyway.
_CHAIN_DEPTH = 20
_CHAIN_COUNT = 1024
_FAN_EPISODE_COUNT = 3
_FAN_STEP_COUNT = 100
_SUBSAMPLE_SEED = 0

# Joins the training stage's wandb panel group.
_LOG_PREFIX = "1_dynamics-diagnostics"


def diagnose_dynamics_model_ensemble(
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

    state_windows, action_windows = _draw_chain_windows(
        validation_episodes=validation_episodes
    )
    state_window_batch = torch.from_numpy(state_windows).to(device=device)
    action_window_batch = torch.from_numpy(action_windows).to(device=device)

    logger.info("Diagnosing on %d held-out chains.", state_window_batch.shape[0])
    forward_chain_measurements = _measure_forward_chains(
        dynamics_model_ensemble=dynamics_model_ensemble,
        state_windows=state_window_batch,
        action_windows=action_window_batch,
    )
    backward_chain_measurements = _measure_backward_chains(
        dynamics_model_ensemble=dynamics_model_ensemble,
        state_windows=state_window_batch,
        action_windows=action_window_batch,
        convergence_tolerance=1e-3 * median_expert_step_distance,
    )

    forward_spearman = _spearman_correlation(
        claims=forward_chain_measurements.disagreements,
        truths=forward_chain_measurements.true_errors,
    )
    backward_spearman = _spearman_correlation(
        claims=backward_chain_measurements.disagreements,
        truths=backward_chain_measurements.true_errors,
    )

    run.summary[f"{_LOG_PREFIX}/spearman_forward"] = forward_spearman
    run.summary[f"{_LOG_PREFIX}/spearman_backward"] = backward_spearman
    run.summary[f"{_LOG_PREFIX}/backward_full_depth_fraction"] = float(
        np.isfinite(backward_chain_measurements.true_errors[:, -1].cpu().numpy()).mean()
    )
    # How many times larger each raw filter signal is than the true error at
    # the same backward step. Translates the swept threshold into true-error
    # units without fixing a signal mix here.
    for signal_name, signals in (
        ("disagreement", backward_chain_measurements.disagreements),
        (
            "lipschitz_distance",
            backward_chain_measurements.sensitivity_scaled_step_distances,
        ),
    ):
        signal_values, truth_values = _finite_pairs(
            claims=signals, truths=backward_chain_measurements.true_errors
        )
        if signal_values.size > 0:
            run.summary[f"{_LOG_PREFIX}/signal_over_truth_{signal_name}"] = float(
                np.median(signal_values / np.maximum(truth_values, 1e-12))
            )

    forward_disagreements, forward_true_errors = _finite_pairs(
        claims=forward_chain_measurements.disagreements,
        truths=forward_chain_measurements.true_errors,
    )
    backward_disagreements, backward_true_errors = _finite_pairs(
        claims=backward_chain_measurements.disagreements,
        truths=backward_chain_measurements.true_errors,
    )
    run.log(
        data={
            f"{_LOG_PREFIX}/error_over_depth": _error_over_depth_panel(
                forward_true_errors_per_depth=(
                    forward_chain_measurements.true_errors.cpu().numpy()
                ),
                backward_true_errors_per_depth=(
                    backward_chain_measurements.true_errors.cpu().numpy()
                ),
            ),
            f"{_LOG_PREFIX}/disagreement_vs_true_error": (
                _disagreement_scatter_panel(
                    forward_disagreements=forward_disagreements,
                    forward_true_errors=forward_true_errors,
                    backward_disagreements=backward_disagreements,
                    backward_true_errors=backward_true_errors,
                )
            ),
            **_member_fan_panels(
                dynamics_model_ensemble=dynamics_model_ensemble,
                validation_episodes=validation_episodes,
                device=device,
            ),
        }
    )


# ---- Private: measurement -------------------------------


@dataclass(frozen=True)
class _ForwardChainMeasurements:
    """Open-loop rollouts of the ensemble mean, one chain per window.

    Entry [c, d] belongs to chain c at depth d + 1: the model has taken d + 1
    steps on its own predictions. All values are in normalized state units.
    """

    # Distance between the model's prediction and the real state.
    true_errors: torch.Tensor  # (chain_count, depth_count)
    # Ensemble disagreement at the state the prediction was made from.
    disagreements: torch.Tensor  # (chain_count, depth_count)


@dataclass(frozen=True)
class _BackwardChainMeasurements:
    """Backward walks from the windows' final states, one chain per window.

    Entry [c, d] belongs to chain c at depth d + 1: the Newton solver has
    stepped d + 1 states back from the anchor. A chain ends at its first
    non-converged solve; entries past that point stay NaN.
    """

    # Distance between the solved predecessor and the real predecessor.
    true_errors: torch.Tensor  # (chain_count, depth_count)
    # The two filter signals, evaluated at the solved predecessor.
    disagreements: torch.Tensor  # (chain_count, depth_count)
    sensitivity_scaled_step_distances: torch.Tensor  # (chain_count, depth_count)


def _draw_chain_windows(
    *, validation_episodes: list[TransitionSet]
) -> tuple[
    npt.NDArray[np.float32],  # (chain_count, _CHAIN_DEPTH + 1, state_dimensionality)
    npt.NDArray[np.float32],  # (chain_count, _CHAIN_DEPTH, action_dimensionality)
]:
    state_windows, action_windows = build_windows(
        episodes=validation_episodes, window_length=_CHAIN_DEPTH
    )

    window_count = state_windows.shape[0]
    if window_count <= _CHAIN_COUNT:
        return state_windows, action_windows

    # The windows overlap heavily (stride 1), so a random subset loses no
    # coverage, only redundancy.
    chain_indices = np.random.default_rng(seed=_SUBSAMPLE_SEED).choice(
        window_count, size=_CHAIN_COUNT, replace=False
    )
    return state_windows[chain_indices], action_windows[chain_indices]


@torch.no_grad()
def _measure_forward_chains(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    state_windows: torch.Tensor,  # (chain_count, depth_count + 1, state_dimensionality)
    action_windows: torch.Tensor,  # (chain_count, depth_count, action_dimensionality)
) -> _ForwardChainMeasurements:
    depth_count = action_windows.shape[1]

    current_state_batch = state_windows[:, 0, :]
    true_error_columns: list[torch.Tensor] = []
    disagreement_columns: list[torch.Tensor] = []

    for depth_index in range(depth_count):
        action_batch = action_windows[:, depth_index, :]
        disagreement_columns.append(
            dynamics_model_ensemble.disagreements_about_next_state(
                state_batch=current_state_batch, action_batch=action_batch
            )
        )
        current_state_batch = dynamics_model_ensemble.predict_next_state(
            state_batch=current_state_batch, action_batch=action_batch
        )
        true_error_columns.append(
            _normalized_distances(
                dynamics_model_ensemble=dynamics_model_ensemble,
                state_batch=current_state_batch,
                other_state_batch=state_windows[:, depth_index + 1, :],
            )
        )

    return _ForwardChainMeasurements(
        true_errors=torch.stack(tensors=true_error_columns, dim=1),
        disagreements=torch.stack(tensors=disagreement_columns, dim=1),
    )


@torch.no_grad()
def _measure_backward_chains(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    state_windows: torch.Tensor,  # (chain_count, depth_count + 1, state_dimensionality)
    action_windows: torch.Tensor,  # (chain_count, depth_count, action_dimensionality)
    convergence_tolerance: float,
) -> _BackwardChainMeasurements:
    chain_count, depth_count, _ = action_windows.shape
    device = state_windows.device

    current_state_batch = state_windows[:, -1, :].clone()
    active_mask_batch = torch.ones(chain_count, dtype=torch.bool, device=device)

    true_errors = torch.full(
        size=(chain_count, depth_count), fill_value=float("nan"), device=device
    )
    disagreements = torch.full_like(true_errors, fill_value=float("nan"))
    sensitivity_scaled_step_distances = torch.full_like(
        true_errors, fill_value=float("nan")
    )

    for depth_index in tqdm(
        iterable=range(depth_count), desc="backward diagnostic chains", unit="depth"
    ):
        active_indices = active_mask_batch.nonzero().squeeze(dim=1)
        if active_indices.numel() == 0:
            break

        # The action that led into the chain's current state, and therefore
        # out of the predecessor the solver is asked for.
        window_index = depth_count - 1 - depth_index
        action_batch = action_windows[active_indices, window_index, :]
        successor_state_batch = current_state_batch[active_indices]

        inverse_dynamics_solution = solve_inverse_dynamics(
            dynamics_model_ensemble=dynamics_model_ensemble,
            initial_state_estimate_batch=successor_state_batch,
            action_batch=action_batch,
            target_next_state_batch=successor_state_batch,
            convergence_tolerance=convergence_tolerance,
        )

        converged_mask_batch = inverse_dynamics_solution.converged_batch
        active_mask_batch[active_indices[~converged_mask_batch]] = False
        converged_indices = active_indices[converged_mask_batch]
        if converged_indices.numel() == 0:
            break

        predecessor_state_batch = inverse_dynamics_solution.state_batch[
            converged_mask_batch
        ]
        converged_action_batch = action_batch[converged_mask_batch]

        true_errors[converged_indices, depth_index] = _normalized_distances(
            dynamics_model_ensemble=dynamics_model_ensemble,
            state_batch=predecessor_state_batch,
            other_state_batch=state_windows[converged_indices, window_index, :],
        )
        disagreements[converged_indices, depth_index] = (
            dynamics_model_ensemble.disagreements_about_next_state(
                state_batch=predecessor_state_batch, action_batch=converged_action_batch
            )
        )
        sensitivity_scaled_step_distances[converged_indices, depth_index] = (
            dynamics_model_ensemble.sensitivity_scaled_step_distances(
                state_batch=predecessor_state_batch,
                next_state_batch=successor_state_batch[converged_mask_batch],
                action_batch=converged_action_batch,
            )
        )

        current_state_batch[converged_indices] = predecessor_state_batch

    return _BackwardChainMeasurements(
        true_errors=true_errors,
        disagreements=disagreements,
        sensitivity_scaled_step_distances=sensitivity_scaled_step_distances,
    )


@torch.no_grad()
def _roll_out_members_forward(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    initial_state: torch.Tensor,  # (state_dimensionality,)
    actions: torch.Tensor,  # (step_count, action_dimensionality)
) -> torch.Tensor:  # (member_count, step_count + 1, state_dimensionality)
    """Every member imagines the same action sequence on its own — the fan
    whose spread shows where the ensemble stops agreeing."""

    member_trajectories: list[torch.Tensor] = []
    for dynamics_model in dynamics_model_ensemble.dynamics_models:
        current_state_batch = initial_state.unsqueeze(dim=0)
        trajectory_states = [current_state_batch]
        for step_index in range(actions.shape[0]):
            current_state_batch = dynamics_model.predict_next_state(
                normalizers=dynamics_model_ensemble.normalizers,
                state_batch=current_state_batch,
                action_batch=actions[step_index].unsqueeze(dim=0),
            )
            trajectory_states.append(current_state_batch)
        member_trajectories.append(torch.cat(tensors=trajectory_states, dim=0))

    return torch.stack(tensors=member_trajectories, dim=0)


def _normalized_distances(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    other_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
) -> torch.Tensor:  # (batch_size,)
    return torch.linalg.norm(
        dynamics_model_ensemble.normalize_state(state_batch=state_batch)
        - dynamics_model_ensemble.normalize_state(state_batch=other_state_batch),
        dim=-1,
    )


def _spearman_correlation(*, claims: torch.Tensor, truths: torch.Tensor) -> float:
    claim_values, truth_values = _finite_pairs(claims=claims, truths=truths)
    if claim_values.size < 2:
        return float("nan")
    return float(scipy.stats.spearmanr(a=claim_values, b=truth_values).statistic)


def _finite_pairs(
    *, claims: torch.Tensor, truths: torch.Tensor
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    claim_values = claims.cpu().numpy().ravel()
    truth_values = truths.cpu().numpy().ravel()
    finite_mask = np.isfinite(claim_values) & np.isfinite(truth_values)
    return claim_values[finite_mask], truth_values[finite_mask]


# ---- Private: panels ------------------------------------


# One fixed color per chain direction, reused across all panels.
_FORWARD_COLOR = "#1f77b4"
_BACKWARD_COLOR = "#d62728"

# PushT's quadratic arena (gym_pusht window size, pixels).
_ARENA_SIZE = 512


def _error_over_depth_panel(
    *,
    forward_true_errors_per_depth: npt.NDArray[
        np.floating
    ],  # (chain_count, depth_count)
    backward_true_errors_per_depth: npt.NDArray[
        np.floating
    ],  # (chain_count, depth_count), NaN allowed
) -> wandb.Image:
    """Backward steps run to the left of 0, forward steps to the right.
    Line = median, band = 10th–90th percentile."""
    figure, axes = plt.subplots(figsize=(9, 4.5))

    for true_errors_per_depth, direction_sign, color in (
        (backward_true_errors_per_depth, -1, _BACKWARD_COLOR),
        (forward_true_errors_per_depth, +1, _FORWARD_COLOR),
    ):
        depth_count = true_errors_per_depth.shape[1]
        step_positions = direction_sign * np.arange(1, depth_count + 1)
        axes.plot(
            step_positions,
            np.nanmedian(true_errors_per_depth, axis=0),
            marker="o",
            markersize=3,
            color=color,
        )
        # Nested percentile bands; their overlap darkens toward the median,
        # so the shading reads as a density gradient.
        for lower_quantile in (10, 20, 30, 40):
            axes.fill_between(
                step_positions,
                np.nanpercentile(true_errors_per_depth, q=lower_quantile, axis=0),
                np.nanpercentile(
                    true_errors_per_depth, q=100 - lower_quantile, axis=0
                ),
                alpha=0.12,
                color=color,
                linewidth=0,
            )

    axes.axvline(x=0, color="black", linewidth=1.0)
    axes.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes.set_xlabel(xlabel="steps (negative = backward)")
    axes.set_ylabel(ylabel="error (normalized)")
    axes.set_yscale(value="log")
    axes.set_title(label="Prediction error vs. number of steps")
    axes.legend(
        handles=[
            plt.Line2D([], [], color=_FORWARD_COLOR, label="forward"),
            plt.Line2D([], [], color=_BACKWARD_COLOR, label="backward"),
            plt.Line2D(
                [], [], color="black", marker="o", markersize=3, label="median"
            ),
            patches.Patch(
                facecolor="gray", alpha=0.35, label="spread (darker = denser)"
            ),
        ],
        fontsize=8,
    )
    figure.tight_layout()

    image = wandb.Image(figure)
    plt.close(fig=figure)
    return image


def _disagreement_scatter_panel(
    *,
    forward_disagreements: npt.NDArray[np.floating],  # (point_count,)
    forward_true_errors: npt.NDArray[np.floating],  # (point_count,)
    backward_disagreements: npt.NDArray[np.floating],  # (point_count,)
    backward_true_errors: npt.NDArray[np.floating],  # (point_count,)
) -> wandb.Image:
    """One dot per chain step. A cloud rising to the right means the
    ensemble's uncertainty tracks its real error."""
    # Backward on the left, forward on the right — same order as the
    # error-over-depth panel.
    figure, (backward_axes, forward_axes) = plt.subplots(
        ncols=2, figsize=(10, 5), sharex=True, sharey=True
    )

    for axes, disagreements, true_errors, direction_label, color in (
        (
            backward_axes,
            backward_disagreements,
            backward_true_errors,
            "backward",
            _BACKWARD_COLOR,
        ),
        (
            forward_axes,
            forward_disagreements,
            forward_true_errors,
            "forward",
            _FORWARD_COLOR,
        ),
    ):
        axes.scatter(disagreements, true_errors, s=4, alpha=0.3, color=color)
        axes.set_title(label=direction_label)
        axes.set_xlabel(xlabel="predicted uncertainty")
        axes.set_xscale(value="log")
        axes.set_yscale(value="log")

    backward_axes.set_ylabel(ylabel="actual error")
    figure.suptitle(t="Predicted uncertainty vs. actual error")
    figure.tight_layout()

    image = wandb.Image(figure)
    plt.close(fig=figure)
    return image


def _member_fan_panels(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    validation_episodes: list[TransitionSet],
    device: torch.device,
) -> dict[str, wandb.Image]:
    fan_panels: dict[str, wandb.Image] = {}
    for episode_index in range(min(_FAN_EPISODE_COUNT, len(validation_episodes))):
        episode = validation_episodes[episode_index]
        step_count = min(_FAN_STEP_COUNT, episode.actions.shape[0])

        true_trajectory = np.concatenate(
            [episode.states[:step_count], episode.next_states[[step_count - 1]]], axis=0
        )  # (step_count + 1, state_dimensionality)

        member_trajectories = _roll_out_members_forward(
            dynamics_model_ensemble=dynamics_model_ensemble,
            initial_state=torch.from_numpy(true_trajectory[0]).to(device=device),
            actions=torch.from_numpy(episode.actions[:step_count]).to(device=device),
        )

        fan_panels[f"{_LOG_PREFIX}/fan_episode_{episode_index}"] = _member_fan_panel(
            member_trajectories=member_trajectories.cpu().numpy(),
            true_trajectory=true_trajectory,
            episode_index=episode_index,
        )
    return fan_panels


def _member_fan_panel(
    *,
    member_trajectories: npt.NDArray[
        np.floating
    ],  # (member_count, step_count + 1, state_dimensionality)
    true_trajectory: npt.NDArray[np.floating],  # (step_count + 1, state_dimensionality)
    episode_index: int,
) -> wandb.Image:
    """The MBPO-style fan: every ensemble model replays the recorded actions on
    its own predictions. Where the thin lines spread, the ensemble is
    uncertain; where they leave the recorded path, it is wrong. State layout is
    PushT: agent x/y, block x/y, angle."""
    figure, (agent_axes, block_axes) = plt.subplots(
        ncols=2, figsize=(10, 5), sharex=True, sharey=True
    )

    for axes, dimension_slice, title in (
        (agent_axes, slice(0, 2), "agent position"),
        (block_axes, slice(2, 4), "block position"),
    ):
        for member_index, member_trajectory in enumerate(
            iterable=member_trajectories
        ):
            path = member_trajectory[:, dimension_slice]
            axes.plot(
                path[:, 0],
                path[:, 1],
                linewidth=0.6,
                alpha=0.35,
                color=_FORWARD_COLOR,
                # One legend entry for all members.
                label="model predictions" if member_index == 0 else None,
            )
        true_path = true_trajectory[:, dimension_slice]
        axes.plot(
            true_path[:, 0],
            true_path[:, 1],
            linewidth=2.0,
            color="black",
            label="recorded episode",
        )
        axes.plot(
            true_path[0, 0],
            true_path[0, 1],
            marker="o",
            markersize=9,
            color="black",
            markerfacecolor="white",
            linestyle="none",
            label="start",
        )
        axes.plot(
            true_path[-1, 0],
            true_path[-1, 1],
            marker="X",
            markersize=9,
            color="black",
            linestyle="none",
            label="end (recorded)",
        )
        axes.set_title(label=title)
        axes.set_xlabel(xlabel="x position")
        axes.set_ylabel(ylabel="y position")
        axes.set_xlim(left=0, right=_ARENA_SIZE)
        axes.set_ylim(bottom=0, top=_ARENA_SIZE)
        # "box" because the two subplots share their axes ranges.
        axes.set_aspect(aspect="equal", adjustable="box")
        # sharey hides the right subplot's tick labels; bring them back.
        axes.tick_params(labelleft=True)

    # One legend for the whole figure; both subplots share the same entries.
    handles, labels = agent_axes.get_legend_handles_labels()
    figure.legend(handles=handles, labels=labels, loc="lower center", ncol=4)
    figure.suptitle(
        t=f"Replay: true dynamics vs. model (held-out episode {episode_index})"
    )
    figure.tight_layout(rect=(0, 0.08, 1, 0.94))

    image = wandb.Image(figure)
    plt.close(fig=figure)
    return image
