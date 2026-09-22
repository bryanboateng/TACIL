import matplotlib
import numpy as np
import wandb
from matplotlib import pyplot as plt
from matplotlib.artist import Artist
from matplotlib.legend_handler import HandlerTuple
from numpy import typing as npt


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
