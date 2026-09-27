import matplotlib
import numpy as np
import wandb
from numpy import typing as npt

from scripts.environments.peg_insertion import observation as obs

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


def create_dynamics_plots(
    *,
    member_trajectories: npt.NDArray[
        np.floating
    ],  # (member_count, step_count, state_dimensionality)
    true_trajectory: npt.NDArray[np.floating],  # (step_count, state_dimensionality)
    episode_index: int,
) -> dict[str, wandb.Image]:
    images = {}
    for name, title, components in _component_groups():
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


def _component_groups() -> list[tuple[str, str, list[tuple[int, str, str, float]]]]:
    return [
        (
            "tcp_and_gripper",
            "TCP and gripper",
            _body_components(
                position=obs.TCP_POSITION_SLICE,
                orientation=obs.TCP_ORIENTATION_SLICE,
                linear_velocity=obs.TCP_LINEAR_VELOCITY_SLICE,
                angular_velocity=obs.TCP_ANGULAR_VELOCITY_SLICE,
            )
            + [(obs.GRIPPER_OPENING_INDEX, "Finger position", "mm", 1000.0)],
        ),
        (
            "peg",
            "Peg",
            _body_components(
                position=obs.PEG_POSITION_SLICE,
                orientation=obs.PEG_ORIENTATION_SLICE,
                linear_velocity=obs.PEG_LINEAR_VELOCITY_SLICE,
                angular_velocity=obs.PEG_ANGULAR_VELOCITY_SLICE,
            ),
        ),
        (
            "hole",
            "Hole",
            _vector_components(obs.HOLE_POSITION_SLICE, "Position", "mm", 1000.0)
            + _rotation_components(obs.HOLE_ORIENTATION_SLICE),
        ),
        (
            "dimensions",
            "Object dimensions",
            _vector_components(obs.PEG_HALF_SIZE_SLICE, "Peg half-size", "mm", 1000.0)
            + [(obs.HOLE_RADIUS_INDEX, "Hole radius", "mm", 1000.0)],
        ),
    ]


def _body_components(
    *,
    position: slice,
    orientation: slice,
    linear_velocity: slice,
    angular_velocity: slice,
) -> list[tuple[int, str, str, float]]:
    return (
        _vector_components(position, "Position", "mm", 1000.0)
        + _rotation_components(orientation)
        + _vector_components(linear_velocity, "Linear velocity", "mm/s", 1000.0)
        + _vector_components(angular_velocity, "Angular velocity", "rad/s", 1.0)
    )


def _vector_components(
    entries: slice, label: str, unit: str, scale: float
) -> list[tuple[int, str, str, float]]:
    return [
        (index, f"{label} {axis}", unit, scale)
        for index, axis in zip(range(entries.start, entries.stop), "xyz", strict=True)
    ]


def _rotation_components(entries: slice) -> list[tuple[int, str, str, float]]:
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
