import matplotlib
import numpy as np
import wandb
from numpy import typing as npt

matplotlib.use(backend="Agg")
from matplotlib import patches  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402

plt.style.use(style="seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300

from scripts.shared.pusht_state import (  # noqa: E402
    ARENA_SIZE,
    BLOCK_ANGLE_INDEX,
    STATE_GROUPS,
)

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


def error_over_depth_panel(
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
    axes.set_title(label="One-step error: training vs. validation")
    axes.legend(fontsize=8)
    figure.tight_layout()

    image = wandb.Image(figure)
    plt.close(fig=figure)
    return image


def uncertainty_scatter_panel(
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
