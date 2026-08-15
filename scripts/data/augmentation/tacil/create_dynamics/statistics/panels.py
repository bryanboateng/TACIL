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
    STATE_GROUPS,
)

_FORWARD_COLOR = "tab:red"
_BACKWARD_COLOR = "tab:blue"


def error_over_depth_panel(
    *,
    forward_errors_per_depth: npt.NDArray[np.floating],  # (chain_count, depth_count)
    backward_errors_per_depth: npt.NDArray[np.floating],  # (chain_count, depth_count)
) -> wandb.Image:
    figure, (backward_subplot, forward_subplot) = plt.subplots(
        ncols=2, figsize=(10, 5), sharey=True
    )

    for subplot, errors_per_depth, direction_label, color, direction_sign in (
        (backward_subplot, backward_errors_per_depth, "Backward", _BACKWARD_COLOR, -1),
        (forward_subplot, forward_errors_per_depth, "Forward", _FORWARD_COLOR, +1),
    ):
        measured_depth_mask = np.isfinite(errors_per_depth).any(axis=0)
        low_percentiles, medians, high_percentiles = np.nanpercentile(
            a=errors_per_depth[:, measured_depth_mask], q=(10, 50, 90), axis=0
        )
        positions = (
            direction_sign
            * np.arange(1, errors_per_depth.shape[1] + 1)[measured_depth_mask]
        )

        drawing_order = np.argsort(positions)

        subplot.plot(
            positions[drawing_order],
            medians[drawing_order],
            color=color,
            linewidth=2.0,
            label="median",
        )
        subplot.fill_between(
            positions[drawing_order],
            low_percentiles[drawing_order],
            high_percentiles[drawing_order],
            color=color,
            alpha=0.15,
            linewidth=0,
            label="10th-90th percentile",
        )

        subplot.legend()
        subplot.set_title(label=direction_label)
        subplot.set_xlabel(xlabel="steps")
        subplot.set_yscale(value="log")
        subplot.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    backward_subplot.set_ylabel(ylabel="error")
    figure.suptitle(t="Prediction Error vs. Number of Steps")
    figure.tight_layout()

    image = wandb.Image(data_or_path=figure)
    plt.close(fig=figure)
    return image


def split_comparison_panel(
    *, training: dict[str, float], validation: dict[str, float]
) -> wandb.Image:
    """Training against validation, per state part. Equal bars mean the model
    generalizes; a much lower training bar means it overfits."""
    group_names = [group_name for group_name, _ in STATE_GROUPS]
    group_positions = np.arange(len(group_names))
    bar_width = 0.38

    one_step_errors_per_split: dict[str, dict[str, float]] = {
        "training": training,
        "validation": validation,
    }

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
    axes.set_title(label="One-Step Error: Training vs. Validation")
    axes.legend(fontsize=8)
    figure.tight_layout()

    image = wandb.Image(figure)
    plt.close(fig=figure)
    return image


def prediction_error_vs_prediction_uncertainty(
    *,
    forward_uncertainties: npt.NDArray[np.floating],  # (point_count,)
    forward_true_errors: npt.NDArray[np.floating],  # (point_count,)
    backward_uncertainties: npt.NDArray[np.floating],  # (point_count,)
    backward_true_errors: npt.NDArray[np.floating],  # (point_count,)
) -> wandb.Image:
    figure, (backward_subplot, forward_subplot) = plt.subplots(ncols=2, figsize=(10, 5))

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
        subplot.set_ylabel(ylabel="error")
        subplot.set_xscale(value="log")
        subplot.set_yscale(value="log")

    figure.suptitle(t="Prediction Error vs. Prediction Uncertainty")
    figure.tight_layout()

    image = wandb.Image(data_or_path=figure)
    plt.close(fig=figure)
    return image


def replay_panel(
    *,
    member_trajectories: npt.NDArray[
        np.floating
    ],  # (member_count, step_count + 1, state_dimensionality)
    true_trajectory: npt.NDArray[np.floating],  # (step_count + 1, state_dimensionality)
    episode_index: int,
) -> wandb.Image:
    (
        figure,
        (agent_position_subplot, block_position_subplot, block_orientation_subplot),
    ) = plt.subplots(ncols=3, figsize=(14, 5))

    prediction_width = 0.8
    prediction_alpha = 0.5
    prediction_color = _FORWARD_COLOR

    a_width = 1.6
    a_color = "black"

    for subplot, dimension_slice, title in (
        (agent_position_subplot, slice(0, 2), "Agent Position"),
        (block_position_subplot, slice(2, 4), "Block Position"),
    ):
        if not isinstance(subplot, plt.Axes):
            raise TypeError()

        for member_index, member_trajectory in enumerate(iterable=member_trajectories):
            path = member_trajectory[:, dimension_slice]
            subplot.plot(
                path[:, 0],
                path[:, 1],
                linewidth=prediction_width,
                alpha=prediction_alpha,
                color=prediction_color,
                # One legend entry for all members.
                label="model predictions" if member_index == 0 else None,
            )
        true_path = true_trajectory[:, dimension_slice]
        subplot.plot(
            true_path[:, 0],
            true_path[:, 1],
            linewidth=a_width,
            color=a_color,
            label="recorded episode",
        )
        subplot.set_title(label=title)
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
        linewidth=a_width,
        color=a_color,
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

    # One legend for the whole figure because the subplots share the same entries.
    handles, labels = agent_position_subplot.get_legend_handles_labels()
    figure.legend(handles=handles, labels=labels)
    figure.suptitle(t=f"True Dynamics vs. Model (Validation Episode {episode_index})")
    figure.tight_layout()

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
