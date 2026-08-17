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
    BLOCK_ORIENTATION_INDEX,
)

_FORWARD_COLOR = "tab:red"
_BACKWARD_COLOR = "tab:blue"


def error_over_depth_panel(
    *,
    forward_errors_per_depth: npt.NDArray[np.floating],  # (chain_count, depth_count)
    backward_errors_per_depth: npt.NDArray[np.floating],  # (chain_count, depth_count)
    forward_disagreements_per_depth: npt.NDArray[
        np.floating
    ],  # (chain_count, depth_count)
    backward_disagreements_per_depth: npt.NDArray[
        np.floating
    ],  # (chain_count, depth_count)
) -> wandb.Image:
    """Error against the number of steps, over the whole state at once.

    Backward chains run to the left of zero, forward chains to the right, so
    both directions share one axis and stay comparable.

    The dashed line is how wrong the ensemble believes it might be, against the
    solid line of how wrong it actually is. Backward, the dashed line is the
    gate's own signal, so the gap between the two is what the gate is working
    with: a dashed line far under the solid one means the gate is told less
    than it needs to stop in time.
    """
    figure, axes = plt.subplots(figsize=(9, 5))

    for (
        errors_per_depth,
        disagreements_per_depth,
        direction_label,
        color,
        direction_sign,
    ) in (
        (
            backward_errors_per_depth,
            backward_disagreements_per_depth,
            "backward",
            _BACKWARD_COLOR,
            -1,
        ),
        (
            forward_errors_per_depth,
            forward_disagreements_per_depth,
            "forward",
            _FORWARD_COLOR,
            +1,
        ),
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

        disagreement_depth_mask = np.isfinite(disagreements_per_depth).any(axis=0)
        if not disagreement_depth_mask.any():
            continue
        disagreement_positions = (
            direction_sign
            * np.arange(1, disagreements_per_depth.shape[1] + 1)[
                disagreement_depth_mask
            ]
        )
        disagreement_order = np.argsort(disagreement_positions)
        axes.plot(
            disagreement_positions[disagreement_order],
            np.nanmedian(a=disagreements_per_depth[:, disagreement_depth_mask], axis=0)[
                disagreement_order
            ],
            color=color,
            linewidth=2.0,
            linestyle="--",
            label=f"{direction_label} ensemble disagreement (median)",
        )

    axes.set_xlabel(xlabel="steps")
    # Not "error": the dashed line is a spread, not a distance from the truth.
    # Both are lengths in the same normalized state space, which is what lets
    # them share the axis.
    axes.set_ylabel(ylabel="normalized state distance")
    axes.set_yscale(value="log")
    axes.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes.legend()
    axes.set_title(label="Prediction Error and Ensemble Disagreement vs. Steps")
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
