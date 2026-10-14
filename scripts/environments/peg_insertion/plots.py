import numpy as np
import wandb
from numpy import typing as npt

from scripts.environments.maniskill.dynamics_plots import (
    ComponentGroup,
    body_components,
    rotation_components,
    vector_components,
)
from scripts.environments.maniskill import dynamics_plots
from scripts.environments.peg_insertion import observation as obs


def create_dynamics_plots(
    *,
    member_trajectories: npt.NDArray[
        np.floating
    ],  # (member_count, step_count, state_dimensionality)
    true_trajectory: npt.NDArray[np.floating],  # (step_count, state_dimensionality)
    episode_index: int,
) -> dict[str, wandb.Image]:
    return dynamics_plots.create_dynamics_plots(
        component_groups=_component_groups(),
        member_trajectories=member_trajectories,
        true_trajectory=true_trajectory,
        episode_index=episode_index,
    )


def _component_groups() -> list[ComponentGroup]:
    return [
        (
            "tcp_and_gripper",
            "TCP and gripper",
            body_components(
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
            body_components(
                position=obs.PEG_POSITION_SLICE,
                orientation=obs.PEG_ORIENTATION_SLICE,
                linear_velocity=obs.PEG_LINEAR_VELOCITY_SLICE,
                angular_velocity=obs.PEG_ANGULAR_VELOCITY_SLICE,
            ),
        ),
        (
            "hole",
            "Hole",
            vector_components(obs.HOLE_POSITION_SLICE, "Position", "mm", 1000.0)
            + rotation_components(obs.HOLE_ORIENTATION_SLICE),
        ),
        (
            "dimensions",
            "Object dimensions",
            vector_components(obs.PEG_HALF_SIZE_SLICE, "Peg half-size", "mm", 1000.0)
            + [(obs.HOLE_RADIUS_INDEX, "Hole radius", "mm", 1000.0)],
        ),
    ]
