import matplotlib
import numpy as np
import wandb
from numpy import typing as npt

matplotlib.use(backend="Agg")
from matplotlib import pyplot as plt  # noqa: E402

plt.style.use(style="seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300

from scripts.shared.pusht_state import (  # noqa: E402
    ARENA_SIZE,
    BLOCK_ANGLE_INDEX,
)

_FORWARD_COLOR = "tab:red"
_BACKWARD_COLOR = "tab:blue"


def error_over_depth_panel(
    *,
    forward_errors_per_depth: npt.NDArray[np.floating],  # (chain_count, depth_count)
    backward_errors_per_depth: npt.NDArray[np.floating],  # (chain_count, depth_count)
) -> wandb.Image:
    """Error against the number of steps, over the whole state at once.

    Backward chains run to the left of zero, forward chains to the right, so
    both directions share one error axis and stay comparable.
    """
    figure, axes = plt.subplots(figsize=(9, 5))

    for errors_per_depth, direction_label, color, direction_sign in (
        (backward_errors_per_depth, "backward", _BACKWARD_COLOR, -1),
        (forward_errors_per_depth, "forward", _FORWARD_COLOR, +1),
    ):
        measured_depth_mask = np.isfinite(errors_per_depth).any(axis=0)
        if not measured_depth_mask.any():
            continue

        low_percentiles, medians, high_percentiles = np.nanpercentile(
            a=errors_per_depth[:, measured_depth_mask], q=(10, 50, 90), axis=0
        )
        positions = (
            direction_sign
            * np.arange(1, errors_per_depth.shape[1] + 1)[measured_depth_mask]
        )
        drawing_order = np.argsort(positions)

        axes.plot(
            positions[drawing_order],
            medians[drawing_order],
            color=color,
            linewidth=2.0,
            label=f"{direction_label} median",
        )
        axes.fill_between(
            positions[drawing_order],
            low_percentiles[drawing_order],
            high_percentiles[drawing_order],
            color=color,
            alpha=0.15,
            linewidth=0,
            label=f"{direction_label} 10th-90th percentile",
        )

    axes.set_xlabel(xlabel="steps")
    axes.set_ylabel(ylabel="error (normalized)")
    axes.set_yscale(value="log")
    axes.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes.legend()
    axes.set_title(label="Prediction Error vs. Number of Steps")
    figure.tight_layout()

    image = wandb.Image(data_or_path=figure)
    plt.close(fig=figure)
    return image


def spearman_over_depth_panel(
    *,
    forward_correlations_per_depth: npt.NDArray[np.floating],  # (depth_count,)
    backward_correlations_per_depth: npt.NDArray[np.floating],  # (depth_count,)
) -> wandb.Image:
    """How well the uncertainty orders the true error, at each depth.

    The gate only needs the ordering, so 1 is perfect and 0 says the
    uncertainty carries no information about the error.
    """
    figure, axes = plt.subplots(figsize=(8, 5))

    for correlations_per_depth, direction_label, color, direction_sign in (
        (backward_correlations_per_depth, "backward", _BACKWARD_COLOR, -1),
        (forward_correlations_per_depth, "forward", _FORWARD_COLOR, +1),
    ):
        measured_depth_mask = np.isfinite(correlations_per_depth)
        if not measured_depth_mask.any():
            continue

        positions = (
            direction_sign
            * np.arange(1, correlations_per_depth.shape[0] + 1)[measured_depth_mask]
        )
        drawing_order = np.argsort(positions)

        axes.plot(
            positions[drawing_order],
            correlations_per_depth[measured_depth_mask][drawing_order],
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=4,
            label=direction_label,
        )

    # No ordering at all, the line the curves must stay above to be useful.
    axes.axhline(y=0.0, color="gray", linewidth=0.8, linestyle="--")
    axes.set_ylim(bottom=-1.0, top=1.0)
    axes.set_xlabel(xlabel="steps")
    axes.set_ylabel(ylabel="Spearman correlation")
    axes.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes.legend()
    axes.set_title(label="Uncertainty vs. Error: Ordering per Depth")
    figure.tight_layout()

    image = wandb.Image(data_or_path=figure)
    plt.close(fig=figure)
    return image


def prediction_error_vs_prediction_uncertainty(
    *,
    forward_uncertainties: npt.NDArray[np.floating],  # (point_count,)
    forward_true_errors: npt.NDArray[np.floating],  # (point_count,)
    backward_uncertainties: npt.NDArray[np.floating],  # (point_count,)
    backward_true_errors: npt.NDArray[np.floating],  # (point_count,)
) -> wandb.Image:
    """Uncertainty against the error it is supposed to stand in for.

    Both directions share one pair of axes. How tightly a cloud rises is only
    half the story — the other half is which errors a direction even reaches,
    and separate axes would hide exactly that.
    """
    figure, (backward_subplot, forward_subplot) = plt.subplots(
        ncols=2, figsize=(10, 5), sharex=True, sharey=True
    )

    for subplot, uncertainties, true_errors, direction_label, color in (
        (
            backward_subplot,
            backward_uncertainties,
            backward_true_errors,
            "Backward",
            _BACKWARD_COLOR,
        ),
        (
            forward_subplot,
            forward_uncertainties,
            forward_true_errors,
            "Forward",
            _FORWARD_COLOR,
        ),
    ):
        subplot.scatter(uncertainties, true_errors, s=4, alpha=0.3, color=color)
        subplot.set_title(label=direction_label)
        subplot.set_xlabel(xlabel="uncertainty")
        subplot.set_xscale(value="log")
        subplot.set_yscale(value="log")

    backward_subplot.set_ylabel(ylabel="error")

    figure.suptitle(t="Prediction Error vs. Prediction Uncertainty")
    figure.tight_layout()

    image = wandb.Image(data_or_path=figure)
    plt.close(fig=figure)
    return image


def replay_panel(
    *,
    forward_member_trajectories: npt.NDArray[
        np.floating
    ],  # (member_count, step_count + 1, state_dimensionality)
    backward_member_trajectories: npt.NDArray[
        np.floating
    ],  # (member_count, step_count + 1, state_dimensionality)
    true_trajectory: npt.NDArray[np.floating],  # (step_count + 1, state_dimensionality)
    episode_index: int,
) -> wandb.Image:
    """One recorded episode, replayed by every member on its own.

    The top row starts at the episode's first state and predicts forward, the
    bottom row starts at its last state and inverts its way back. A member that
    stops converging leaves its row early.
    """
    figure, subplot_rows = plt.subplots(nrows=2, ncols=3, figsize=(14, 9))

    prediction_width = 0.8
    prediction_alpha = 0.5

    recorded_width = 1.6
    recorded_color = "black"

    for subplots, member_trajectories, prediction_color, direction_label in (
        (subplot_rows[0], forward_member_trajectories, _FORWARD_COLOR, "forward"),
        (subplot_rows[1], backward_member_trajectories, _BACKWARD_COLOR, "backward"),
    ):
        agent_position_subplot, block_position_subplot, block_orientation_subplot = (
            subplots
        )

        for subplot, dimension_slice, title in (
            (agent_position_subplot, slice(0, 2), "Agent Position"),
            (block_position_subplot, slice(2, 4), "Block Position"),
        ):
            for member_index, member_trajectory in enumerate(
                iterable=member_trajectories
            ):
                path = member_trajectory[:, dimension_slice]
                subplot.plot(
                    path[:, 0],
                    path[:, 1],
                    linewidth=prediction_width,
                    alpha=prediction_alpha,
                    color=prediction_color,
                    # One legend entry for all members.
                    label=f"{direction_label} predictions"
                    if member_index == 0
                    else None,
                )
            true_path = true_trajectory[:, dimension_slice]
            subplot.plot(
                true_path[:, 0],
                true_path[:, 1],
                linewidth=recorded_width,
                color=recorded_color,
                label="recorded episode",
            )
            subplot.set_title(label=f"{title} ({direction_label})")
            subplot.set_xlabel(xlabel="x position [px]")
            subplot.set_ylabel(ylabel="y position [px]")
            subplot.set_xlim(left=0, right=ARENA_SIZE)
            subplot.set_ylim(bottom=0, top=ARENA_SIZE)
            subplot.set_aspect(aspect="equal", adjustable="box")

        steps = np.arange(true_trajectory.shape[0])
        for member_trajectory in member_trajectories:
            block_orientation_subplot.plot(
                steps,
                _break_at_wraps(angle_timeline=member_trajectory[:, BLOCK_ANGLE_INDEX]),
                linewidth=prediction_width,
                alpha=prediction_alpha,
                color=prediction_color,
            )
        block_orientation_subplot.plot(
            steps,
            _break_at_wraps(angle_timeline=true_trajectory[:, BLOCK_ANGLE_INDEX]),
            linewidth=recorded_width,
            color=recorded_color,
        )
        block_orientation_subplot.set_title(
            label=f"Block Orientation ({direction_label})"
        )
        block_orientation_subplot.set_xlabel(xlabel="step")
        block_orientation_subplot.set_ylabel(ylabel="orientation [rad]")
        block_orientation_subplot.set_ylim(bottom=0, top=2.0 * np.pi)
        block_orientation_subplot.yaxis.set_major_locator(
            locator=plt.MultipleLocator(base=np.pi / 2)
        )
        block_orientation_subplot.yaxis.set_major_formatter(
            formatter=plt.FuncFormatter(func=_pi_multiple_label)
        )

    # One legend for the whole figure, with the recorded episode kept once —
    # both rows draw it, and the two rows differ only in their prediction color.
    handles = []
    labels = []
    for row_subplots in subplot_rows:
        for handle, label in zip(*row_subplots[0].get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    figure.legend(handles=handles, labels=labels, loc="lower center", ncol=3)
    figure.suptitle(t=f"True Dynamics vs. Model (Validation Episode {episode_index})")
    figure.tight_layout(rect=(0, 0.05, 1, 1))

    image = wandb.Image(data_or_path=figure)
    plt.close(fig=figure)
    return image


# ---- Private --------------------------------------------


def _pi_multiple_label(value: float, _position: int) -> str:
    """Format a tick that `MultipleLocator(pi / 2)` placed, e.g. `pi`, `3pi/2`."""
    quarter_turns = round(value / (np.pi / 2))
    if quarter_turns == 0:
        return "0"
    if quarter_turns % 2 == 0:
        half_turns = quarter_turns // 2
        return "π" if half_turns == 1 else f"{half_turns}π"
    return "π/2" if quarter_turns == 1 else f"{quarter_turns}π/2"


def _break_at_wraps(
    *,
    angle_timeline: npt.NDArray[np.floating],  # (step_count,)
) -> npt.NDArray[np.floating]:  # (step_count,)
    """NaN out the step right after a 0/2π crossing, so the plotted line lifts
    the pen there instead of drawing a spurious jump across the full range."""
    broken_timeline = angle_timeline.copy()
    crossed_seam = np.abs(np.diff(a=angle_timeline)) > np.pi
    broken_timeline[1:][crossed_seam] = np.nan
    return broken_timeline
