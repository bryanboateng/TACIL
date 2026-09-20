import matplotlib
import numpy as np
import wandb
from matplotlib import pyplot as plt
from matplotlib.artist import Artist
from matplotlib.legend_handler import HandlerTuple
from numpy import typing as npt

from scripts.environments.plot_layout import PlotLayout

matplotlib.use(backend="Agg")

plt.style.use(style="seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300

_FORWARD_COLOR = "tab:red"
_BACKWARD_COLOR = "tab:blue"
_REPLAY_ERROR_COLOR = "tab:green"

_PANEL_WIDTH = 3.9
_PANEL_HEIGHT = 4.5


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
    ],  # (window_count, step_count)
    backward_replay_errors_per_step_count: npt.NDArray[
        np.floating
    ],  # (window_count, step_count)
) -> wandb.Image:
    figure, plot_area = plt.subplots(figsize=(9, 5))

    legend_handles: list[tuple[Artist, Artist]] = []
    legend_labels: list[str] = []

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
    plot_area.set_xlim(left=0, right=float(positions.min()) - 0.5)
    plot_area.legend(
        handles=legend_handles,
        labels=legend_labels,
        title="median and 10th-90th percentile",
        handler_map={tuple: HandlerTuple(ndivide=1)},
    )
    plot_area.set_title(label="Replay Error vs. Prediction Error")
    figure.tight_layout()

    image = wandb.Image(data_or_path=figure)
    plt.close(fig=figure)
    return image


def create_true_dynamics_vs_model_plot(
    *,
    member_trajectories: npt.NDArray[
        np.floating
    ],  # (member_count, step_count + 1, state_dimensionality)
    true_trajectory: npt.NDArray[np.floating],  # (step_count + 1, state_dimensionality)
    plot_layout: PlotLayout,
    episode_index: int,
) -> wandb.Image:
    position_groups = plot_layout.position_groups
    orientation_groups = plot_layout.orientation_groups
    subplot_count = len(position_groups) + len(orientation_groups)

    position_scale = plot_layout.position_unit.scale
    horizontal_limits = plot_layout.horizontal_limits
    vertical_limits = plot_layout.vertical_limits

    position_width_ratio = (horizontal_limits[1] - horizontal_limits[0]) / (
        vertical_limits[1] - vertical_limits[0]
    )
    width_ratios = [position_width_ratio] * len(position_groups) + [1.0] * len(
        orientation_groups
    )

    figure, subplots = plt.subplots(
        nrows=1,
        ncols=subplot_count,
        figsize=(_PANEL_WIDTH * sum(width_ratios), _PANEL_HEIGHT),
        squeeze=False,
        gridspec_kw={"width_ratios": width_ratios},
        layout="constrained",
    )
    position_subplots = subplots[0, : len(position_groups)]
    orientation_subplots = subplots[0, len(position_groups) :]

    prediction_line_width = 0.8
    prediction_line_alpha = 0.5
    prediction_line_color = _FORWARD_COLOR

    true_line_width = 1.6
    true_line_color = "black"

    for subplot, position_group in zip(position_subplots, position_groups, strict=True):
        for member_index, member_trajectory in enumerate(iterable=member_trajectories):
            path = member_trajectory[:, position_group.entries] * position_scale
            subplot.plot(
                path[:, 0],
                path[:, 1],
                linewidth=prediction_line_width,
                alpha=prediction_line_alpha,
                color=prediction_line_color,
                label="model prediction" if member_index == 0 else None,
            )
        true_path = true_trajectory[:, position_group.entries] * position_scale
        subplot.plot(
            true_path[:, 0],
            true_path[:, 1],
            linewidth=true_line_width,
            color=true_line_color,
            label="recorded episode",
        )
        subplot.set_title(label=position_group.label)
        subplot.set_xlabel(xlabel=f"x position [{plot_layout.position_unit.label}]")
        subplot.set_ylabel(ylabel=f"y position [{plot_layout.position_unit.label}]")
        subplot.set_xlim(left=horizontal_limits[0], right=horizontal_limits[1])
        subplot.set_ylim(bottom=vertical_limits[0], top=vertical_limits[1])
        subplot.set_aspect(aspect="equal", adjustable="box")

    steps = np.arange(true_trajectory.shape[0])
    for subplot, orientation_group in zip(
        orientation_subplots, orientation_groups, strict=True
    ):
        for member_trajectory in member_trajectories:
            subplot.plot(
                steps,
                _break_at_wraps(
                    orientation_timeline=member_trajectory[:, orientation_group.index]
                ),
                linewidth=prediction_line_width,
                alpha=prediction_line_alpha,
                color=prediction_line_color,
            )
        subplot.plot(
            steps,
            _break_at_wraps(
                orientation_timeline=true_trajectory[:, orientation_group.index]
            ),
            linewidth=true_line_width,
            color=true_line_color,
        )
        subplot.set_title(label=orientation_group.label)
        subplot.set_xlabel(xlabel="step")
        subplot.set_ylabel(ylabel="orientation [rad]")
        subplot.set_ylim(bottom=0, top=2.0 * np.pi)
        subplot.yaxis.set_major_locator(locator=plt.MultipleLocator(base=np.pi / 2))
        subplot.yaxis.set_major_formatter(
            formatter=plt.FuncFormatter(func=_describe_pi_multiple)
        )
        subplot.set_box_aspect(aspect=1.0)

    figure.legend(
        *position_subplots[0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=2,
    )
    figure.suptitle(t=f"True Dynamics vs. Model (Validation Episode {episode_index})")

    image = wandb.Image(data_or_path=figure)
    plt.close(fig=figure)
    return image


# ---- Private --------------------------------------------


def _describe_pi_multiple(value: float, _position: int) -> str:
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
    broken_timeline = orientation_timeline.copy()
    crossed_seam = np.abs(np.diff(a=orientation_timeline)) > np.pi
    broken_timeline[1:][crossed_seam] = np.nan
    return broken_timeline
