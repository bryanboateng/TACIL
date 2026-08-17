import matplotlib
import numpy as np
import wandb
from matplotlib import pyplot as plt
from matplotlib.artist import Artist
from matplotlib.legend_handler import HandlerTuple
from numpy import typing as npt

from scripts.shared.pusht_state import ARENA_SIZE, BLOCK_ORIENTATION_INDEX

matplotlib.use(backend="Agg")

plt.style.use(style="seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300

_FORWARD_COLOR = "tab:red"
_BACKWARD_COLOR = "tab:blue"
_REPLAY_ERROR_COLOR = "tab:green"


def create_prediction_error_vs_step_count_plot(
    *,
    forward_prediction_errors_per_step_count: npt.NDArray[
        np.floating
    ],  # (trajectory_count, step_count)
    backward_prediction_errors_per_step_count: npt.NDArray[
        np.floating
    ],  # (trajectory_count, step_count)
) -> wandb.Image:
    figure, plot_area = plt.subplots(figsize=(9, 5))

    legend_handles: list[tuple[Artist, Artist]] = []
    legend_labels: list[str] = []

    for prediction_errors_per_step_count, direction_label, color, direction_sign in (
        (backward_prediction_errors_per_step_count, "backward", _BACKWARD_COLOR, -1),
        (forward_prediction_errors_per_step_count, "forward", _FORWARD_COLOR, +1),
    ):
        measured_step_count_mask = np.isfinite(prediction_errors_per_step_count).any(
            axis=0
        )
        if not measured_step_count_mask.any():
            continue

        low_percentiles, medians, high_percentiles = np.nanpercentile(
            a=prediction_errors_per_step_count[:, measured_step_count_mask],
            q=(10, 50, 90),
            axis=0,
        )
        positions = (
            direction_sign
            * np.arange(1, prediction_errors_per_step_count.shape[1] + 1)[
                measured_step_count_mask
            ]
        )
        drawing_order = np.argsort(positions)

        median_line = plot_area.plot(
            positions[drawing_order], medians[drawing_order], color=color, linewidth=2.0
        )[0]
        percentile_band = plot_area.fill_between(
            positions[drawing_order],
            low_percentiles[drawing_order],
            high_percentiles[drawing_order],
            color=color,
            alpha=0.15,
            linewidth=0,
        )

        legend_handles.append((percentile_band, median_line))
        legend_labels.append(direction_label)

    plot_area.set_xlabel(xlabel="steps")
    plot_area.set_ylabel(ylabel="error (normalized)")
    plot_area.set_yscale(value="log")
    plot_area.xaxis.set_major_locator(locator=plt.MaxNLocator(integer=True))
    plot_area.legend(
        handles=legend_handles,
        labels=legend_labels,
        title="median and 10th-90th percentile",
        handler_map={tuple: HandlerTuple(ndivide=1)},
    )
    plot_area.set_title(label="Prediction Error vs. Number of Steps")
    figure.tight_layout()

    image = wandb.Image(data_or_path=figure)
    plt.close(fig=figure)
    return image


def create_replay_error_plot(
    *,
    backward_prediction_errors_per_step_count: npt.NDArray[
        np.floating
    ],  # (chain_count, step_count)
    backward_replay_errors_per_step_count: npt.NDArray[
        np.floating
    ],  # (chain_count, step_count)
) -> wandb.Image:
    figure, plot_area = plt.subplots(figsize=(9, 5))

    legend_handles: list[tuple[Artist, Artist]] = []
    legend_labels: list[str] = []

    # Negative, like the backward half of the error plot, so a step count keeps
    # the same sign in both. The axis is flipped below, which puts the anchor
    # at the left and lets the chain grow rightwards as it is read.
    positions = -np.arange(1, backward_prediction_errors_per_step_count.shape[1] + 1)
    for values_per_step_count, label, color, line_style in (
        (
            backward_prediction_errors_per_step_count,
            "prediction error",
            _BACKWARD_COLOR,
            "-",
        ),
        (
            backward_replay_errors_per_step_count,
            "replay error",
            _REPLAY_ERROR_COLOR,
            "--",
        ),
    ):
        measured_step_count_mask = np.isfinite(values_per_step_count).any(axis=0)
        if not measured_step_count_mask.any():
            continue

        low_percentiles, medians, high_percentiles = np.nanpercentile(
            a=values_per_step_count[:, measured_step_count_mask], q=(10, 50, 90), axis=0
        )
        drawing_order = np.argsort(positions[measured_step_count_mask])
        median_line = plot_area.plot(
            positions[measured_step_count_mask][drawing_order],
            medians[drawing_order],
            color=color,
            linewidth=2.0,
            linestyle=line_style,
        )[0]
        percentile_band = plot_area.fill_between(
            positions[measured_step_count_mask][drawing_order],
            low_percentiles[drawing_order],
            high_percentiles[drawing_order],
            color=color,
            alpha=0.15,
            linewidth=0,
        )

        legend_handles.append((percentile_band, median_line))
        legend_labels.append(label)

    plot_area.set_xlabel(xlabel="steps")
    plot_area.set_ylabel(ylabel="normalized state distance")
    plot_area.set_yscale(value="log")
    plot_area.xaxis.set_major_locator(locator=plt.MaxNLocator(integer=True))
    # Zero on the left, the deepest step on the right, so the chain grows in
    # the direction the plot is read.
    plot_area.set_xlim(left=0, right=float(positions.min()) - 0.5)
    plot_area.legend(
        handles=legend_handles,
        labels=legend_labels,
        title="median and 10th-90th percentile",
        handler_map={tuple: HandlerTuple(ndivide=1)},
    )
    plot_area.set_title(label="Replay Error vs. True Error")
    figure.tight_layout()

    image = wandb.Image(data_or_path=figure)
    plt.close(fig=figure)
    return image


def replay_panel(
    *,
    forward_member_trajectories: npt.NDArray[
        np.floating
    ],  # (member_count, step_count + 1, state_dimensionality)
    true_trajectory: npt.NDArray[np.floating],  # (step_count + 1, state_dimensionality)
    episode_index: int,
) -> wandb.Image:
    """One recorded episode, replayed forward by every member on its own.

    Only forward. The backward map is not a function -- many predecessors lead
    to the same successor -- so a backward replay spreads out however well the
    model fits, and says nothing about its quality.
    """
    figure, subplots = plt.subplots(nrows=1, ncols=3, figsize=(14, 4.5))

    prediction_width = 0.8
    prediction_alpha = 0.5

    recorded_width = 1.6
    recorded_color = "black"

    agent_position_subplot, block_position_subplot, block_orientation_subplot = subplots

    for subplot, dimension_slice, title in (
        (agent_position_subplot, slice(0, 2), "Agent Position"),
        (block_position_subplot, slice(2, 4), "Block Position"),
    ):
        for member_index, member_trajectory in enumerate(
            iterable=forward_member_trajectories
        ):
            path = member_trajectory[:, dimension_slice]
            subplot.plot(
                path[:, 0],
                path[:, 1],
                linewidth=prediction_width,
                alpha=prediction_alpha,
                color=_FORWARD_COLOR,
                # One legend entry for all members.
                label="model prediction" if member_index == 0 else None,
            )
        true_path = true_trajectory[:, dimension_slice]
        subplot.plot(
            true_path[:, 0],
            true_path[:, 1],
            linewidth=recorded_width,
            color=recorded_color,
            label="recorded episode",
        )
        subplot.set_title(label=title)
        subplot.set_xlabel(xlabel="x position [px]")
        subplot.set_ylabel(ylabel="y position [px]")
        subplot.set_xlim(left=0, right=ARENA_SIZE)
        subplot.set_ylim(bottom=0, top=ARENA_SIZE)
        subplot.set_aspect(aspect="equal", adjustable="box")

    steps = np.arange(true_trajectory.shape[0])
    for member_trajectory in forward_member_trajectories:
        block_orientation_subplot.plot(
            steps,
            _break_at_wraps(
                orientation_timeline=member_trajectory[:, BLOCK_ORIENTATION_INDEX]
            ),
            linewidth=prediction_width,
            alpha=prediction_alpha,
            color=_FORWARD_COLOR,
        )
    block_orientation_subplot.plot(
        steps,
        _break_at_wraps(
            orientation_timeline=true_trajectory[:, BLOCK_ORIENTATION_INDEX]
        ),
        linewidth=recorded_width,
        color=recorded_color,
    )
    block_orientation_subplot.set_title(label="Block Orientation")
    block_orientation_subplot.set_xlabel(xlabel="step")
    block_orientation_subplot.set_ylabel(ylabel="orientation [rad]")
    block_orientation_subplot.set_ylim(bottom=0, top=2.0 * np.pi)
    block_orientation_subplot.yaxis.set_major_locator(
        locator=plt.MultipleLocator(base=np.pi / 2)
    )
    block_orientation_subplot.yaxis.set_major_formatter(
        formatter=plt.FuncFormatter(func=_pi_multiple_label)
    )

    # One legend for the figure, taken from the first subplot -- all three
    # draw the same two kinds of line.
    figure.legend(*subplots[0].get_legend_handles_labels(), loc="lower center", ncol=2)
    figure.suptitle(t=f"True Dynamics vs. Model (Validation Episode {episode_index})")
    figure.tight_layout(rect=(0, 0.08, 1, 1))

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
    orientation_timeline: npt.NDArray[np.floating],  # (step_count,)
) -> npt.NDArray[np.floating]:  # (step_count,)
    """NaN out the step right after a 0/2π crossing, so the plotted line lifts
    the pen there instead of drawing a spurious jump across the full range."""
    broken_timeline = orientation_timeline.copy()
    crossed_seam = np.abs(np.diff(a=orientation_timeline)) > np.pi
    broken_timeline[1:][crossed_seam] = np.nan
    return broken_timeline
