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
    """Every ensemble model replays the recorded actions on its own
    predictions. Where the thin lines spread, the ensemble is uncertain;
    where they leave the recorded path, it is wrong. State layout is PushT:
    agent x/y, block x/y, angle."""
    figure, (agent_axes, block_axes, angle_axes) = plt.subplots(
        ncols=3, figsize=(14, 5)
    )

    for axes, dimension_slice, title in (
        (agent_axes, slice(0, 2), "Agent Position"),
        (block_axes, slice(2, 4), "Block Position"),
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
    angle_axes.set_title(label="Block Orientation")
    angle_axes.set_xlabel(xlabel="step")
    angle_axes.set_ylabel(ylabel="turned since start (rad)")

    # One legend for the whole figure; the subplots share the same entries.
    handles, labels = agent_axes.get_legend_handles_labels()
    figure.legend(handles=handles, labels=labels, loc="lower center", ncol=4)
    figure.suptitle(
        t=f"Replay: True Dynamics vs. Model (Held-Out Episode {episode_index})"
    )
    figure.tight_layout(rect=(0, 0.08, 1, 0.94))

    image = wandb.Image(figure)
    plt.close(fig=figure)
    return image


# ---- Private --------------------------------------------


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
