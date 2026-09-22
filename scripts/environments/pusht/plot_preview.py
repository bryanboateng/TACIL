import numpy as np
from numpy import typing as npt

from scripts.environments.pusht import observation as obs

_FULL_TURN = 2.0 * np.pi
_DRIFT_FRACTION = 0.06


def draw_trajectories(
    *, rng: np.random.Generator, step_count: int, member_count: int
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    progress = np.linspace(0.0, 1.0, step_count)

    true_trajectory = np.zeros(
        shape=(step_count, obs.STATE_DIMENSIONALITY), dtype=np.float64
    )
    drift_per_entry = np.zeros(obs.STATE_DIMENSIONALITY)

    for group_index, entries in enumerate(
        (obs.AGENT_POSITION_SLICE, obs.BLOCK_POSITION_SLICE)
    ):
        phase = 0.12 * group_index
        true_trajectory[:, entries] = np.stack(
            arrays=[
                obs.ARENA_SIZE * (0.12 + 0.76 * progress),
                obs.ARENA_SIZE
                * (0.12 + 0.76 * np.sin(np.pi * (progress + phase)) ** 2),
            ],
            axis=-1,
        )  # (step_count, 2)
        drift_per_entry[entries] = _DRIFT_FRACTION * np.array(
            object=[obs.ARENA_SIZE, obs.ARENA_SIZE]
        )

    true_trajectory[:, obs.BLOCK_ORIENTATION_INDEX] = (
        1.25 * _FULL_TURN * progress
    ) % _FULL_TURN
    drift_per_entry[obs.BLOCK_ORIENTATION_INDEX] = _DRIFT_FRACTION * _FULL_TURN

    member_trajectories = true_trajectory[None] + (
        progress[None, :, None] ** 1.5
        * rng.normal(
            loc=0.0, scale=1.0, size=(member_count, 1, obs.STATE_DIMENSIONALITY)
        )
        * drift_per_entry
    )  # (member_count, step_count, state_dimensionality)
    member_trajectories[..., obs.BLOCK_ORIENTATION_INDEX] %= _FULL_TURN

    return member_trajectories, true_trajectory
