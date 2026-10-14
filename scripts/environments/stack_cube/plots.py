import numpy as np
import wandb
from numpy import typing as npt

from scripts.environments.maniskill import dynamics_plots
from scripts.environments.maniskill.dynamics_plots import (
    ComponentGroup,
    body_components,
)
from scripts.environments.stack_cube import observation as obs


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
            "cube_a",
            "Cube A (picked)",
            body_components(
                position=obs.CUBE_A_POSITION_SLICE,
                orientation=obs.CUBE_A_ORIENTATION_SLICE,
                linear_velocity=obs.CUBE_A_LINEAR_VELOCITY_SLICE,
                angular_velocity=obs.CUBE_A_ANGULAR_VELOCITY_SLICE,
            ),
        ),
        (
            "cube_b",
            "Cube B (base)",
            body_components(
                position=obs.CUBE_B_POSITION_SLICE,
                orientation=obs.CUBE_B_ORIENTATION_SLICE,
                linear_velocity=obs.CUBE_B_LINEAR_VELOCITY_SLICE,
                angular_velocity=obs.CUBE_B_ANGULAR_VELOCITY_SLICE,
            ),
        ),
    ]
