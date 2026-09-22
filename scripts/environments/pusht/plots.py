import matplotlib
import numpy as np
import wandb
from numpy import typing as npt

from scripts.environments.pusht.observation import (
    AGENT_POSITION_SLICE,
    BLOCK_POSITION_SLICE,
    BLOCK_ORIENTATION_INDEX,
    ARENA_SIZE,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt

_PANEL_WIDTH = 3.9
_PANEL_HEIGHT = 4.5
_FORWARD_COLOR = "tab:red"


def create_dynamics_plots(
    *,
    member_trajectories: npt.NDArray[
        np.floating
    ],  # (member_count, step_count + 1, state_dimensionality)
    true_trajectory: npt.NDArray[np.floating],  # (step_count + 1, state_dimensionality)
    episode_index: int,
) -> dict[str, wandb.Image]:
    figure, subplots = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(3 * _PANEL_WIDTH, _PANEL_HEIGHT),
        layout="constrained",
    )

    prediction_line_width = 0.8
    prediction_line_alpha = 0.5
    prediction_line_color = _FORWARD_COLOR

    true_line_width = 1.6
    true_line_color = "black"

    for subplot, label, entries in (
        (subplots[0], "Agent Position", AGENT_POSITION_SLICE),
        (subplots[1], "Block Position", BLOCK_POSITION_SLICE),
    ):
        for member_index, member_trajectory in enumerate(iterable=member_trajectories):
            path = member_trajectory[:, entries]
            subplot.plot(
                path[:, 0],
                path[:, 1],
                linewidth=prediction_line_width,
                alpha=prediction_line_alpha,
                color=prediction_line_color,
                label="model prediction" if member_index == 0 else None,
            )
        true_path = true_trajectory[:, entries]
        subplot.plot(
            true_path[:, 0],
            true_path[:, 1],
            linewidth=true_line_width,
            color=true_line_color,
            label="recorded episode",
        )
        subplot.set_title(label=label)
        subplot.set_xlabel(xlabel="x position [px]")
        subplot.set_ylabel(ylabel="y position [px]")
        subplot.set_xlim(left=0, right=ARENA_SIZE)
        subplot.set_ylim(bottom=0, top=ARENA_SIZE)
        subplot.set_aspect(aspect="equal", adjustable="box")

    steps = np.arange(true_trajectory.shape[0])
    subplot = subplots[2]
    for member_trajectory in member_trajectories:
        subplot.plot(
            steps,
            _break_at_wraps(
                orientation_timeline=member_trajectory[:, BLOCK_ORIENTATION_INDEX]
            ),
            linewidth=prediction_line_width,
            alpha=prediction_line_alpha,
            color=prediction_line_color,
        )
    subplot.plot(
        steps,
        _break_at_wraps(
            orientation_timeline=true_trajectory[:, BLOCK_ORIENTATION_INDEX]
        ),
        linewidth=true_line_width,
        color=true_line_color,
    )
    subplot.set_title(label="Block Orientation")
    subplot.set_xlabel(xlabel="step")
    subplot.set_ylabel(ylabel="orientation [rad]")
    subplot.set_ylim(bottom=0, top=2.0 * np.pi)
    subplot.yaxis.set_major_locator(locator=plt.MultipleLocator(base=np.pi / 2))
    subplot.yaxis.set_major_formatter(
        formatter=plt.FuncFormatter(func=_describe_pi_multiple)
    )
    subplot.set_box_aspect(aspect=1.0)

    figure.legend(
        *subplots[0].get_legend_handles_labels(), loc="outside lower center", ncol=2
    )
    figure.suptitle(t=f"True Dynamics vs. Model (Validation Episode {episode_index})")

    image = wandb.Image(data_or_path=figure)
    plt.close(fig=figure)
    return {"state": image}


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
