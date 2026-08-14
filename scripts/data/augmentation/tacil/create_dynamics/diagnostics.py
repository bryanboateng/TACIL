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
from matplotlib.legend_handler import HandlerTuple  # noqa: E402

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
from scripts.shared.distribution_summary import summarize_distribution  # noqa: E402
from scripts.shared.episode_store import EpisodeStore  # noqa: E402
from scripts.shared.log import create_logger  # noqa: E402
from scripts.shared.pusht_state import (  # noqa: E402
    ARENA_SIZE,
    BLOCK_ANGLE_INDEX,
    STATE_GROUPS,
    subtract_states,
)

logger = create_logger()

# Fixed measurement shape: deep enough to show the growth trend, small enough
# to stay affordable, and heavily overlapping windows make more chains
# redundant anyway.
_CHAIN_DEPTH = 20
_CHAIN_COUNT = 1024
_REPLAY_EPISODE_COUNT = 3
_REPLAY_STEP_COUNT = 100
_SUBSAMPLE_SEED = 0

# Joins the training stage's wandb panel group.
_STAGE_NAME = "1_dynamics"


def diagnose_dynamics_model_ensemble(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    training_episodes: list[TransitionSet],
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
        # The distance-to-data column compares against the TRAINING data: the
        # question it answers is whether the models had material to learn
        # from, and the models never saw the validation split.
        training_state_batch=torch.from_numpy(
            np.concatenate([episode.states for episode in training_episodes], axis=0)
        ).to(device=device),
    )

    # Measured on the backward chains, which is the direction the augmentation
    # walks: the signal the data stage gates on, and the model-free distance to
    # the training data as the comparison point.
    backward_signals = (
        ("uncertainty", backward_chain_measurements.uncertainties),
        ("training_data_distance", backward_chain_measurements.training_data_distances),
    )
    signals_per_direction = (
        (
            "forward",
            forward_chain_measurements.true_errors,
            (("uncertainty", forward_chain_measurements.uncertainties),),
        ),
        ("backward", backward_chain_measurements.true_errors, backward_signals),
    )

    # Whether a signal ORDERS the true errors correctly. That is what a
    # threshold needs; being off by a constant factor does not matter, since
    # the limit is fitted anyway.
    for direction_name, true_errors, signals in signals_per_direction:
        for signal_name, claims in signals:
            run.summary[f"{_STAGE_NAME}/6_spearman/{direction_name}/{signal_name}"] = (
                _spearman_correlation(claims=claims, truths=true_errors)
            )

            # Pooled over all depths, the growth of the error with depth is the
            # loudest thing in the data. Any signal that also grows with depth
            # then scores high without ordering anything at a fixed depth. Per
            # depth that confound is gone, so this is the number the signal
            # choice rests on.
            correlations_per_depth = _spearman_correlations_per_depth(
                claims=claims, truths=true_errors
            )
            measured_correlations = correlations_per_depth[
                np.isfinite(correlations_per_depth)
            ]
            if measured_correlations.size == 0:
                continue
            for label, value in summarize_distribution(
                values=measured_correlations
            ).items():
                run.summary[
                    f"{_STAGE_NAME}/6_spearman_per_depth/"
                    f"{direction_name}/{signal_name}/{label}"
                ] = value

    # The single-step error per state part on both splits: equal values mean
    # the model is as good as it can be on this data, a training value far
    # below the validation one means it memorized instead of generalized.
    one_step_errors_per_split = {
        split_name: _one_step_group_errors(
            dynamics_model_ensemble=dynamics_model_ensemble, episodes=episodes
        )
        for split_name, episodes in (
            ("training", training_episodes),
            ("validation", validation_episodes),
        )
    }
    for split_name, group_errors in one_step_errors_per_split.items():
        for group_name, group_error in group_errors.items():
            run.summary[f"{_STAGE_NAME}/5_one_step_error/{split_name}/{group_name}"] = (
                group_error
            )
    run.summary[f"{_STAGE_NAME}/7_backward/full_depth_fraction"] = float(
        np.isfinite(backward_chain_measurements.true_errors[:, -1].cpu().numpy()).mean()
    )

    for direction_name, measurements in (
        ("forward", forward_chain_measurements),
        ("backward", backward_chain_measurements),
    ):
        for depth in (1, 5, 10):
            if depth > measurements.true_errors.shape[1]:
                continue
            errors_at_depth = measurements.true_errors[:, depth - 1].cpu().numpy()
            # A chain that stopped before this depth is NaN from there on, and
            # it has no error to contribute.
            errors_at_depth = errors_at_depth[np.isfinite(errors_at_depth)]
            if errors_at_depth.size == 0:
                continue
            for label, value in summarize_distribution(values=errors_at_depth).items():
                run.summary[
                    f"{_STAGE_NAME}/8_error_over_depth/"
                    f"{direction_name}/depth_{depth}/{label}"
                ] = value
    # How many times larger each raw filter signal is than the true error at
    # the same backward step. Says what part of the swept limit range a signal
    # needs, and how far the two are apart in scale.
    for signal_name, signals in backward_signals:
        signal_values, truth_values = _finite_pairs(
            claims=signals, truths=backward_chain_measurements.true_errors
        )
        ratios = signal_values / np.maximum(truth_values, 1e-12)
        for label, value in summarize_distribution(values=ratios).items():
            run.summary[
                f"{_STAGE_NAME}/7_backward/signal_over_truth/{signal_name}/{label}"
            ] = value

    forward_uncertainties, forward_true_errors = _finite_pairs(
        claims=forward_chain_measurements.uncertainties,
        truths=forward_chain_measurements.true_errors,
    )
    backward_uncertainties, backward_true_errors = _finite_pairs(
        claims=backward_chain_measurements.uncertainties,
        truths=backward_chain_measurements.true_errors,
    )
    run.log(
        data={
            f"{_STAGE_NAME}/9_panels/error_over_depth": _error_over_depth_panel(
                forward_true_errors_per_depth=(
                    forward_chain_measurements.true_errors.cpu().numpy()
                ),
                backward_true_errors_per_depth=(
                    backward_chain_measurements.true_errors.cpu().numpy()
                ),
            ),
            f"{_STAGE_NAME}/9_panels/one_step_error_by_split": _split_comparison_panel(
                one_step_errors_per_split=one_step_errors_per_split
            ),
            f"{_STAGE_NAME}/9_panels/uncertainty_vs_error": (
                _uncertainty_scatter_panel(
                    forward_uncertainties=forward_uncertainties,
                    forward_true_errors=forward_true_errors,
                    backward_uncertainties=backward_uncertainties,
                    backward_true_errors=backward_true_errors,
                )
            ),
            **_replay_panels(
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
    # The same deviation kept per dimension of the current frame.
    state_differences: torch.Tensor  # (chain_count, depth_count, state_dimensionality)
    # Ensemble uncertainty at the state the prediction was made from.
    uncertainties: torch.Tensor  # (chain_count, depth_count)


@dataclass(frozen=True)
class _BackwardChainMeasurements:
    """Backward walks from the windows' final states, one chain per window.

    Entry [c, d] belongs to chain c at depth d + 1: the Newton solver has
    stepped d + 1 states back from the anchor. A chain ends at its first
    non-converged solve; entries past that point stay NaN.
    """

    # Distance between the solved predecessor and the real predecessor.
    true_errors: torch.Tensor  # (chain_count, depth_count)
    # The same deviation kept per dimension of the current frame.
    state_differences: torch.Tensor  # (chain_count, depth_count, state_dimensionality)
    # The gate signal and the reference column, evaluated at the solved
    # predecessor.
    uncertainties: torch.Tensor  # (chain_count, depth_count)
    training_data_distances: torch.Tensor  # (chain_count, depth_count)


@torch.no_grad()
def _one_step_group_errors(
    *, dynamics_model_ensemble: DynamicsModelEnsemble, episodes: list[TransitionSet]
) -> dict[str, float]:
    """Median error per state part after a single forward step — the model's
    quality without any compounding mixed in."""
    device = next(dynamics_model_ensemble.parameters()).device
    state_windows, action_windows = _subsample(
        *build_windows(episodes=episodes, window_length=1)
    )

    measurements = _measure_forward_chains(
        dynamics_model_ensemble=dynamics_model_ensemble,
        state_windows=torch.from_numpy(state_windows).to(device=device),
        action_windows=torch.from_numpy(action_windows).to(device=device),
    )
    differences = (
        measurements.state_differences[:, 0, :].cpu().numpy()
    )  # (chain_count, state_dimensionality)

    return {
        group_name: float(
            np.median(np.linalg.norm(differences[:, group_slice], axis=-1))
        )
        for group_name, group_slice in STATE_GROUPS
    }


def _draw_chain_windows(
    *, validation_episodes: list[TransitionSet]
) -> tuple[
    npt.NDArray[np.float32],  # (chain_count, _CHAIN_DEPTH + 1, state_dimensionality)
    npt.NDArray[np.float32],  # (chain_count, _CHAIN_DEPTH, action_dimensionality)
]:
    return _subsample(
        *build_windows(episodes=validation_episodes, window_length=_CHAIN_DEPTH)
    )


def _subsample(
    state_windows: npt.NDArray[np.float32],  # (window_count, length + 1, state_dim)
    action_windows: npt.NDArray[np.float32],  # (window_count, length, action_dim)
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Cap the number of windows. They overlap heavily (stride 1), so a random
    subset loses no coverage, only redundancy."""
    window_count = state_windows.shape[0]
    if window_count <= _CHAIN_COUNT:
        return state_windows, action_windows

    window_indices = np.random.default_rng(seed=_SUBSAMPLE_SEED).choice(
        window_count, size=_CHAIN_COUNT, replace=False
    )
    return state_windows[window_indices], action_windows[window_indices]


@torch.no_grad()
def _measure_forward_chains(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    state_windows: torch.Tensor,  # (chain_count, depth_count + 1, state_dimensionality)
    action_windows: torch.Tensor,  # (chain_count, depth_count, action_dimensionality)
) -> _ForwardChainMeasurements:
    depth_count = action_windows.shape[1]

    current_state_batch = state_windows[:, 0, :]
    difference_columns: list[torch.Tensor] = []
    uncertainty_columns: list[torch.Tensor] = []

    for depth_index in range(depth_count):
        action_batch = action_windows[:, depth_index, :]
        uncertainty_columns.append(
            dynamics_model_ensemble.uncertainty_about_next_state(
                state_batch=current_state_batch, action_batch=action_batch
            )
        )
        current_state_batch = dynamics_model_ensemble.predict_next_state(
            state_batch=current_state_batch, action_batch=action_batch
        )
        difference_columns.append(
            _normalized_differences(
                dynamics_model_ensemble=dynamics_model_ensemble,
                state_batch=current_state_batch,
                other_state_batch=state_windows[:, depth_index + 1, :],
            )
        )

    state_differences = torch.stack(tensors=difference_columns, dim=1)
    return _ForwardChainMeasurements(
        true_errors=torch.linalg.norm(state_differences, dim=-1),
        state_differences=state_differences,
        uncertainties=torch.stack(tensors=uncertainty_columns, dim=1),
    )


@torch.no_grad()
def _measure_backward_chains(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    state_windows: torch.Tensor,  # (chain_count, depth_count + 1, state_dimensionality)
    action_windows: torch.Tensor,  # (chain_count, depth_count, action_dimensionality)
    convergence_tolerance: float,
    training_state_batch: torch.Tensor,  # (training_state_count, state_dimensionality)
) -> _BackwardChainMeasurements:
    chain_count, depth_count, _ = action_windows.shape
    device = state_windows.device

    current_state_batch = state_windows[:, -1, :].clone()
    active_mask_batch = torch.ones(chain_count, dtype=torch.bool, device=device)

    true_errors = torch.full(
        size=(chain_count, depth_count), fill_value=float("nan"), device=device
    )
    state_differences = torch.full(
        size=(chain_count, depth_count, state_windows.shape[-1]),
        fill_value=float("nan"),
        device=device,
    )
    uncertainties = torch.full_like(true_errors, fill_value=float("nan"))
    training_data_distances = torch.full_like(true_errors, fill_value=float("nan"))

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

        difference_batch = _normalized_differences(
            dynamics_model_ensemble=dynamics_model_ensemble,
            state_batch=predecessor_state_batch,
            other_state_batch=state_windows[converged_indices, window_index, :],
        )
        state_differences[converged_indices, depth_index] = difference_batch
        true_errors[converged_indices, depth_index] = torch.linalg.norm(
            difference_batch, dim=-1
        )
        uncertainties[converged_indices, depth_index] = (
            dynamics_model_ensemble.uncertainty_about_next_state(
                state_batch=predecessor_state_batch, action_batch=converged_action_batch
            )
        )
        training_data_distances[converged_indices, depth_index] = (
            _nearest_state_distances(
                dynamics_model_ensemble=dynamics_model_ensemble,
                state_batch=predecessor_state_batch,
                reference_state_batch=training_state_batch,
            )
        )

        current_state_batch[converged_indices] = predecessor_state_batch

    return _BackwardChainMeasurements(
        true_errors=true_errors,
        state_differences=state_differences,
        uncertainties=uncertainties,
        training_data_distances=training_data_distances,
    )


@torch.no_grad()
def _nearest_state_distances(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    reference_state_batch: torch.Tensor,  # (reference_count, state_dimensionality)
) -> torch.Tensor:  # (batch_size,)
    """Distance to the closest reference state, in spread units. A model-free
    measure of how far a state lies from the data the models learned from."""
    nearest_distance_columns: list[torch.Tensor] = []
    # Chunked: the full pairwise difference tensor would not fit in memory.
    for state_chunk in state_batch.split(split_size=256, dim=0):
        difference_batch = (
            dynamics_model_ensemble.normalizers.scale_to_state_spread_units(
                state_difference_batch=subtract_states(
                    minuend_batch=state_chunk.unsqueeze(dim=1),
                    subtrahend_batch=reference_state_batch.unsqueeze(dim=0),
                )
            )
        )  # (chunk_size, reference_count, state_dimensionality)
        nearest_distance_columns.append(
            torch.linalg.norm(difference_batch, dim=-1).amin(dim=1)
        )
    return torch.cat(tensors=nearest_distance_columns)


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


def _rotation_since_start(
    *,
    angle_series: npt.NDArray[np.floating],  # (step_count,)
) -> npt.NDArray[np.floating]:  # (step_count,)
    """Total turning relative to the first step, counted through the 0 / 2π
    seam.

    For plotting only. Everywhere else an orientation stays inside [0, 2π), but
    a curve that drops back to zero on every full turn hides the trend the plot
    exists to show.
    """
    continuous_angle_series = np.unwrap(p=angle_series)
    return continuous_angle_series - continuous_angle_series[0]


def _normalized_differences(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    other_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
) -> torch.Tensor:  # (batch_size, state_dimensionality)
    """How far the two states are apart, per dimension, in units of that
    dimension's spread."""
    return dynamics_model_ensemble.normalizers.scale_to_state_spread_units(
        state_difference_batch=subtract_states(
            minuend_batch=state_batch, subtrahend_batch=other_state_batch
        )
    )


def _spearman_correlation(*, claims: torch.Tensor, truths: torch.Tensor) -> float:
    claim_values, truth_values = _finite_pairs(claims=claims, truths=truths)
    if claim_values.size < 2:
        return float("nan")
    return float(scipy.stats.spearmanr(a=claim_values, b=truth_values).statistic)


def _spearman_correlations_per_depth(
    *,
    claims: torch.Tensor,  # (chain_count, depth_count)
    truths: torch.Tensor,  # (chain_count, depth_count)
) -> npt.NDArray[np.floating]:  # (depth_count,)
    """One correlation per depth. A depth where too few chains survived has no
    correlation to report and stays NaN."""
    return np.array(
        [
            _spearman_correlation(
                claims=claims[:, depth_index], truths=truths[:, depth_index]
            )
            for depth_index in range(claims.shape[1])
        ]
    )


def _finite_pairs(
    *, claims: torch.Tensor, truths: torch.Tensor
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    claim_values = claims.cpu().numpy().ravel()
    truth_values = truths.cpu().numpy().ravel()
    finite_mask = np.isfinite(claim_values) & np.isfinite(truth_values)
    return claim_values[finite_mask], truth_values[finite_mask]


# ---- Private: panels ------------------------------------


# One fixed color per chain direction, reused across all panels. Categorical
# slots 1 and 2: a direction is an identity, not a value and not a status, so
# neither may be red — red is reserved for "something is wrong". Validated as a
# pair against the surface below (worst CVD dE 24.7, normal-vision dE 33.6).
_FORWARD_COLOR = "#2a78d6"
_BACKWARD_COLOR = "#eb6834"

_SURFACE_COLOR = "#fcfcfb"
# Ink, never a series color: identity comes from the mark beside the text.
_TEXT_COLOR = "#0b0b0b"
_MUTED_TEXT_COLOR = "#52514e"
_GRID_COLOR = "#e3e3e0"

# The band is a wash, not a block. The data is the only loud thing here.
_BAND_OPACITY = 0.10


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
    Line = median, band = 10th-90th percentile."""
    figure, axes = plt.subplots(figsize=(9, 4.5))
    figure.patch.set_facecolor(_SURFACE_COLOR)
    axes.set_facecolor(_SURFACE_COLOR)

    for true_errors_per_depth, direction_sign, color, label_alignment in (
        (backward_true_errors_per_depth, -1, _BACKWARD_COLOR, "left"),
        (forward_true_errors_per_depth, +1, _FORWARD_COLOR, "right"),
    ):
        depth_count = true_errors_per_depth.shape[1]
        step_positions = direction_sign * np.arange(1, depth_count + 1)

        # A chain that stopped before a depth is NaN from there on and has no
        # error to contribute. A depth where every chain stopped has nothing
        # to plot at all, so it drops out of the curve.
        measured_depth_mask = np.isfinite(true_errors_per_depth).any(axis=0)
        low_percentiles, medians, high_percentiles = np.nanpercentile(
            true_errors_per_depth[:, measured_depth_mask], q=(10, 50, 90), axis=0
        )

        # Left to right, so the band does not fold back on itself.
        measured_positions = step_positions[measured_depth_mask]
        ascending_order = np.argsort(measured_positions)
        positions = measured_positions[ascending_order]
        medians = medians[ascending_order]

        axes.fill_between(
            positions,
            low_percentiles[ascending_order],
            high_percentiles[ascending_order],
            color=color,
            alpha=_BAND_OPACITY,
            linewidth=0,
        )
        axes.plot(
            positions,
            medians,
            color=color,
            linewidth=2.0,
            solid_capstyle="round",
            solid_joinstyle="round",
        )

        # Only the deepest step is marked and labelled. That is where the
        # chain has compounded the most, so it is the number the plot is read
        # for; a dot and a value on all twenty depths would be noise.
        deepest_index = 0 if direction_sign < 0 else len(positions) - 1
        axes.plot(
            positions[deepest_index],
            medians[deepest_index],
            marker="o",
            markersize=6,
            color=color,
            markeredgecolor=_SURFACE_COLOR,
            markeredgewidth=2,
        )
        axes.annotate(
            text=f"{medians[deepest_index]:.2g}",
            xy=(positions[deepest_index], medians[deepest_index]),
            xytext=(0, 10),
            textcoords="offset points",
            horizontalalignment=label_alignment,
            fontsize=9,
            color=_MUTED_TEXT_COLOR,
        )

    axes.axvline(x=0, color=_GRID_COLOR, linewidth=1.0)
    axes.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes.set_xlabel(xlabel="steps", color=_MUTED_TEXT_COLOR)
    axes.set_ylabel(ylabel="error", color=_MUTED_TEXT_COLOR)
    axes.set_yscale(value="log")
    axes.set_title(
        label="Prediction error vs. number of steps", color=_TEXT_COLOR, fontsize=12
    )
    axes.grid(visible=True, color=_GRID_COLOR, linewidth=0.8, linestyle="-")
    axes.set_axisbelow(b=True)
    axes.tick_params(colors=_MUTED_TEXT_COLOR, labelsize=9)
    for spine in axes.spines.values():
        spine.set_visible(False)
    # Room for the end labels, which sit above the deepest points.
    axes.margins(x=0.04)

    # Each entry stacks the band swatch under the median line, so one row
    # names a direction and shows both marks that belong to it. The legend
    # title says what those two marks mean, which saves repeating "median"
    # and "10th-90th percentile" per direction.
    # Backward first: it is also the left half of the plot.
    legend = axes.legend(
        handles=[
            (
                patches.Patch(facecolor=color, alpha=_BAND_OPACITY, linewidth=0),
                plt.Line2D([], [], color=color, linewidth=2.0),
            )
            for color in (_BACKWARD_COLOR, _FORWARD_COLOR)
        ],
        labels=["backward", "forward"],
        handler_map={tuple: HandlerTuple(ndivide=None)},
        title="median, 10th-90th percentile",
        alignment="left",
        fontsize=9,
        title_fontsize=9,
        loc="upper center",
        frameon=False,
        labelcolor=_MUTED_TEXT_COLOR,
    )
    legend.get_title().set_color(_MUTED_TEXT_COLOR)
    figure.tight_layout()

    image = wandb.Image(figure)
    plt.close(fig=figure)
    return image


def _split_comparison_panel(
    *, one_step_errors_per_split: dict[str, dict[str, float]]
) -> wandb.Image:
    """Training against validation, per state part. Equal bars mean the model
    generalizes; a much lower training bar means it overfits."""
    group_names = [group_name for group_name, _ in STATE_GROUPS]
    group_positions = np.arange(len(group_names))
    bar_width = 0.38

    figure, axes = plt.subplots(figsize=(6.5, 4))
    for offset, (split_name, group_errors) in zip(
        (-bar_width / 2, bar_width / 2), one_step_errors_per_split.items(), strict=True
    ):
        axes.bar(
            group_positions + offset,
            [group_errors[group_name] for group_name in group_names],
            width=bar_width,
            label=split_name,
        )

    axes.set_xticks(ticks=group_positions, labels=group_names)
    axes.set_ylabel(ylabel="median one-step error (normalized)")
    axes.set_yscale(value="log")
    axes.set_title(label="One-step error: training vs. validation")
    axes.legend(fontsize=8)
    figure.tight_layout()

    image = wandb.Image(figure)
    plt.close(fig=figure)
    return image


def _uncertainty_scatter_panel(
    *,
    forward_uncertainties: npt.NDArray[np.floating],  # (point_count,)
    forward_true_errors: npt.NDArray[np.floating],  # (point_count,)
    backward_uncertainties: npt.NDArray[np.floating],  # (point_count,)
    backward_true_errors: npt.NDArray[np.floating],  # (point_count,)
) -> wandb.Image:
    """One dot per chain step. A cloud rising to the right means the
    ensemble's uncertainty tracks its real error."""
    # Backward on the left, forward on the right — same order as the
    # error-over-depth panel.
    figure, (backward_axes, forward_axes) = plt.subplots(
        ncols=2, figsize=(10, 5), sharex=True, sharey=True
    )

    for axes, uncertainties, true_errors, direction_label, color in (
        (
            backward_axes,
            backward_uncertainties,
            backward_true_errors,
            "backward",
            _BACKWARD_COLOR,
        ),
        (
            forward_axes,
            forward_uncertainties,
            forward_true_errors,
            "forward",
            _FORWARD_COLOR,
        ),
    ):
        axes.scatter(uncertainties, true_errors, s=4, alpha=0.3, color=color)
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


def _replay_panels(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    validation_episodes: list[TransitionSet],
    device: torch.device,
) -> dict[str, wandb.Image]:
    replay_panels: dict[str, wandb.Image] = {}
    for episode_index in range(min(_REPLAY_EPISODE_COUNT, len(validation_episodes))):
        episode = validation_episodes[episode_index]
        step_count = min(_REPLAY_STEP_COUNT, episode.actions.shape[0])

        true_trajectory = np.concatenate(
            [episode.states[:step_count], episode.next_states[[step_count - 1]]], axis=0
        )  # (step_count + 1, state_dimensionality)

        member_trajectories = _roll_out_members_forward(
            dynamics_model_ensemble=dynamics_model_ensemble,
            initial_state=torch.from_numpy(true_trajectory[0]).to(device=device),
            actions=torch.from_numpy(episode.actions[:step_count]).to(device=device),
        )

        replay_panels[f"{_STAGE_NAME}/9_panels/replay_episode_{episode_index}"] = (
            _replay_panel(
                member_trajectories=member_trajectories.cpu().numpy(),
                true_trajectory=true_trajectory,
                episode_index=episode_index,
            )
        )
    return replay_panels


def _replay_panel(
    *,
    member_trajectories: npt.NDArray[
        np.floating
    ],  # (member_count, step_count + 1, state_dimensionality)
    true_trajectory: npt.NDArray[np.floating],  # (step_count + 1, state_dimensionality)
    episode_index: int,
) -> wandb.Image:
    """Every ensemble model replays the recorded actions on its own
    predictions. Where the thin lines spread, the ensemble is uncertain;
    where they leave the recorded path, it is wrong. State layout is PushT:
    agent x/y, block x/y, angle."""
    figure, (agent_axes, block_axes, angle_axes) = plt.subplots(
        ncols=3, figsize=(14, 5)
    )

    for axes, dimension_slice, title in (
        (agent_axes, slice(0, 2), "agent position"),
        (block_axes, slice(2, 4), "block position"),
    ):
        for member_index, member_trajectory in enumerate(iterable=member_trajectories):
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
            label="end",
        )
        axes.set_title(label=title)
        axes.set_xlabel(xlabel="x position")
        axes.set_ylabel(ylabel="y position")
        axes.set_xlim(left=0, right=ARENA_SIZE)
        axes.set_ylim(bottom=0, top=ARENA_SIZE)
        axes.set_aspect(aspect="equal", adjustable="box")

    # How far the block has turned since the episode started, rather than the
    # angle itself: the zero of an angle is arbitrary, turning is what the
    # model has to get right, and a rotation past a full turn just keeps
    # counting instead of falling off the axis.
    steps = np.arange(true_trajectory.shape[0])
    recorded_rotation = _rotation_since_start(
        angle_series=true_trajectory[:, BLOCK_ANGLE_INDEX]
    )
    for member_trajectory in member_trajectories:
        angle_axes.plot(
            steps,
            _rotation_since_start(angle_series=member_trajectory[:, BLOCK_ANGLE_INDEX]),
            linewidth=0.6,
            alpha=0.35,
            color=_FORWARD_COLOR,
        )
    angle_axes.plot(steps, recorded_rotation, linewidth=2.0, color="black")
    angle_axes.plot(
        steps[0],
        recorded_rotation[0],
        marker="o",
        markersize=9,
        color="black",
        markerfacecolor="white",
        linestyle="none",
    )
    angle_axes.plot(
        steps[-1],
        recorded_rotation[-1],
        marker="X",
        markersize=9,
        color="black",
        linestyle="none",
    )
    angle_axes.axhline(y=0.0, color="gray", linewidth=0.8, linestyle="--")
    angle_axes.set_title(label="block orientation")
    angle_axes.set_xlabel(xlabel="step")
    angle_axes.set_ylabel(ylabel="turned since start (rad)")

    # One legend for the whole figure; the subplots share the same entries.
    handles, labels = agent_axes.get_legend_handles_labels()
    figure.legend(handles=handles, labels=labels, loc="lower center", ncol=4)
    figure.suptitle(
        t=f"Replay: true dynamics vs. model (held-out episode {episode_index})"
    )
    figure.tight_layout(rect=(0, 0.08, 1, 0.94))

    image = wandb.Image(figure)
    plt.close(fig=figure)
    return image
