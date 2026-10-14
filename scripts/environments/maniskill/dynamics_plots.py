import matplotlib
import numpy as np
import wandb
from numpy import typing as npt

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

Component = tuple[int, str, str, float]  # state index, label, unit, scale
ComponentGroup = tuple[str, str, list[Component]]  # name, title, components


def create_dynamics_plots(
    *,
    component_groups: list[ComponentGroup],
    member_trajectories: npt.NDArray[
        np.floating
    ],  # (member_count, step_count, state_dimensionality)
    true_trajectory: npt.NDArray[np.floating],  # (step_count, state_dimensionality)
    episode_index: int,
) -> dict[str, wandb.Image]:
    images = {}
    for name, title, components in component_groups:
        figure, axes = plt.subplots(
            nrows=(len(components) + 2) // 3,
            ncols=3,
            figsize=(12, 2.3 * ((len(components) + 2) // 3) + 0.8),
            squeeze=False,
            layout="constrained",
        )
        steps = np.arange(len(true_trajectory))
        for axis, (index, label, unit, scale) in zip(axes.flat, components):
            axis.plot(
                steps,
                member_trajectories[:, :, index].T * scale,
                color="tab:red",
                alpha=0.35,
                linewidth=0.8,
            )
            axis.plot(
                steps, true_trajectory[:, index] * scale, color="black", linewidth=1.5
            )
            axis.set_title(label)
            axis.set_xlabel("step")
            axis.set_ylabel(unit)
            axis.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            axis.ticklabel_format(axis="y", style="plain", useOffset=False)
            axis.grid(alpha=0.25)
        for axis in list(axes.flat)[len(components) :]:
            axis.set_visible(False)
        figure.suptitle(f"{title} — Validation Episode {episode_index}")
        figure.legend(
            handles=[
                Line2D([], [], color="black", label="recorded episode"),
                Line2D([], [], color="tab:red", label="model members"),
            ],
            loc="outside lower center",
            ncol=2,
        )
        images[name] = wandb.Image(figure)
        plt.close(figure)
    return images


def body_components(
    *,
    position: slice,
    orientation: slice,
    linear_velocity: slice,
    angular_velocity: slice,
) -> list[Component]:
    return (
        vector_components(position, "Position", "mm", 1000.0)
        + rotation_components(orientation)
        + vector_components(linear_velocity, "Linear velocity", "mm/s", 1000.0)
        + vector_components(angular_velocity, "Angular velocity", "rad/s", 1.0)
    )


def vector_components(
    entries: slice, label: str, unit: str, scale: float
) -> list[Component]:
    return [
        (index, f"{label} {axis}", unit, scale)
        for index, axis in zip(range(entries.start, entries.stop), "xyz", strict=True)
    ]


def rotation_components(entries: slice) -> list[Component]:
    return [
        (
            entries.start + column * 3 + row,
            f"Rotation column {column + 1}: {axis}",
            "dimensionless",
            1.0,
        )
        for column in range(2)
        for row, axis in enumerate("xyz")
    ]
