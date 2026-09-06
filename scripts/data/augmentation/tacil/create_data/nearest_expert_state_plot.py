import matplotlib
import numpy as np
import wandb
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator
from numpy import typing as npt

matplotlib.use(backend="Agg")

plt.style.use(style="seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300


_EXPERT_COLOR = "C1"
_SYNTHETIC_COLOR = "C0"


def create_nearest_expert_state_plot(
    *,
    synthetic_state_distances: npt.NDArray[np.floating],  # (synthetic_count,)
    synthetic_action_distances: npt.NDArray[np.floating],  # (synthetic_count,)
    expert_state_distances: npt.NDArray[np.floating],  # (expert_count,)
    expert_action_distances: npt.NDArray[np.floating],  # (expert_count,)
) -> wandb.Image:
    figure, plot_area = plt.subplots(figsize=(6, 6))

    plot_area.set_xscale(value="log")
    plot_area.set_yscale(value="log")
    plot_area.set_xlim(
        left=_limits(first=synthetic_state_distances, second=expert_state_distances)
    )
    plot_area.set_ylim(
        bottom=_limits(first=synthetic_action_distances, second=expert_action_distances)
    )

    for axis in (plot_area.xaxis, plot_area.yaxis):
        axis.set_major_locator(LogLocator(base=2.0))
        axis.set_major_formatter(FuncFormatter(func=_format_ratio))
        axis.set_minor_locator(NullLocator())

    for state_distances, action_distances, color in (
        (expert_state_distances, expert_action_distances, _EXPERT_COLOR),
        (synthetic_state_distances, synthetic_action_distances, _SYNTHETIC_COLOR),
    ):
        plot_area.scatter(
            x=state_distances,
            y=action_distances,
            s=4,
            alpha=0.4,
            linewidths=0,
            color=color,
        )

    plot_area.set_title(label="Distance to the Nearest Expert State")
    plot_area.set_xlabel(xlabel="state distance [median per step]")
    plot_area.set_ylabel(ylabel="action distance [median per step]")
    plot_area.legend(
        handles=[
            Line2D(xdata=[], ydata=[], marker="o", linestyle="none", color=color)
            for color in (_SYNTHETIC_COLOR, _EXPERT_COLOR)
        ],
        labels=["synthetic steps", "expert steps"],
    )
    figure.tight_layout()

    image = wandb.Image(figure)
    plt.close(fig=figure)
    return image


# ---- Private --------------------------------------------


def _format_ratio(value: float, _position: int) -> str:
    return f"{value:g}"


def _limits(
    *,
    first: npt.NDArray[np.floating],  # (first_count,)
    second: npt.NDArray[np.floating],  # (second_count,)
) -> tuple[float, float]:
    values = np.concatenate([first, second])
    positive_values = values[values > 0.0]
    return (
        float(np.percentile(a=positive_values, q=0.5)),
        float(positive_values.max()),
    )
