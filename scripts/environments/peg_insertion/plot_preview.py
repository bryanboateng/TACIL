import numpy as np
from numpy import typing as npt

from scripts.environments.peg_insertion import observation as obs


def draw_trajectories(
    *, rng: np.random.Generator, step_count: int, member_count: int
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    progress = np.linspace(0.0, 1.0, step_count)
    states = np.zeros((step_count, obs.STATE_DIMENSIONALITY))
    drift = np.full(obs.STATE_DIMENSIONALITY, 0.02)
    for position, rotation, linear, angular, offset in (
        (
            obs.TCP_POSITION_SLICE,
            obs.TCP_ORIENTATION_SLICE,
            obs.TCP_LINEAR_VELOCITY_SLICE,
            obs.TCP_ANGULAR_VELOCITY_SLICE,
            0.0,
        ),
        (
            obs.PEG_POSITION_SLICE,
            obs.PEG_ORIENTATION_SLICE,
            obs.PEG_LINEAR_VELOCITY_SLICE,
            obs.PEG_ANGULAR_VELOCITY_SLICE,
            0.05,
        ),
    ):
        states[:, position] = np.column_stack(
            (
                0.15 * progress + offset,
                -0.2 + 0.1 * np.sin(np.pi * progress),
                0.04 + 0.08 * np.sin(np.pi * progress),
            )
        )
        angle = 0.5 + progress
        states[:, rotation] = np.column_stack(
            (
                np.cos(angle),
                np.sin(angle),
                np.zeros(step_count),
                -np.sin(angle),
                np.cos(angle),
                np.zeros(step_count),
            )
        )
        states[:, linear] = np.gradient(states[:, position], 0.05, axis=0)
        states[:, angular] = [0.0, 0.0, 0.2]
        drift[rotation] = 0.15
        drift[angular] = 0.1
    states[:, obs.GRIPPER_OPENING_INDEX] = 0.04 - 0.03 * progress
    drift[obs.GRIPPER_OPENING_INDEX] = 0.003
    states[:, obs.HOLE_POSITION_SLICE] = [0.2, -0.2, 0.1]
    states[:, obs.HOLE_ORIENTATION_SLICE] = [0.0, 1.0, 0.0, -1.0, 0.0, 0.0]
    states[:, obs.PEG_HALF_SIZE_SLICE] = [0.1, 0.02, 0.02]
    states[:, obs.HOLE_RADIUS_INDEX] = 0.025
    drift[obs.PEG_HALF_SIZE_SLICE] = 0.002
    drift[obs.HOLE_RADIUS_INDEX] = 0.002
    members = states[None] + (
        progress[None, :, None] ** 1.5
        * rng.normal(size=(member_count, 1, obs.STATE_DIMENSIONALITY))
        * drift
    )
    return members, states
