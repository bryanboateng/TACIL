"""Tells whether an imagined state ends the episode.

The dynamics model predicts the next state but not whether the robot fell
over. These rules decide that, and they are the ones OfflineRL-Kit uses in
offlinerlkit/utils/termination_fns.py. They read the state directly, so no
simulator is needed.
"""

from collections.abc import Callable

import numpy as np
from numpy import typing as npt

TerminationFunction = Callable[[npt.NDArray[np.float32]], npt.NDArray[np.bool_]]


def _terminate_halfcheetah(
    state_batch: npt.NDArray[np.float32],  # (batch_size, state_dimensionality)
) -> npt.NDArray[np.bool_]:  # (batch_size,)
    # The half cheetah cannot fall over, so only a state that ran away counts.
    return ~(
        np.all(state_batch > -100.0, axis=-1) & np.all(state_batch < 100.0, axis=-1)
    )


def _terminate_hopper(
    state_batch: npt.NDArray[np.float32],  # (batch_size, state_dimensionality)
) -> npt.NDArray[np.bool_]:  # (batch_size,)
    height_batch = state_batch[:, 0]
    angle_batch = state_batch[:, 1]
    # The reference writes np.abs(next_obs[:, 1:] < 100), which takes the
    # absolute value of the comparison rather than of the value, and so leaves
    # the comparison one sided. Kept as it stands, because the published
    # numbers were produced with it.
    return ~(
        np.isfinite(state_batch).all(axis=-1)
        & (state_batch[:, 1:] < 100.0).all(axis=-1)
        & (height_batch > 0.7)
        & (np.abs(angle_batch) < 0.2)
    )


def _terminate_walker2d(
    state_batch: npt.NDArray[np.float32],  # (batch_size, state_dimensionality)
) -> npt.NDArray[np.bool_]:  # (batch_size,)
    height_batch = state_batch[:, 0]
    angle_batch = state_batch[:, 1]
    return ~(
        np.all(state_batch > -100.0, axis=-1)
        & np.all(state_batch < 100.0, axis=-1)
        & (height_batch > 0.8)
        & (height_batch < 2.0)
        & (angle_batch > -1.0)
        & (angle_batch < 1.0)
    )


_TERMINATION_FUNCTIONS: dict[str, TerminationFunction] = {
    "halfcheetah": _terminate_halfcheetah,
    "hopper": _terminate_hopper,
    "walker2d": _terminate_walker2d,
}


def get_termination_function(*, robot: str) -> TerminationFunction:
    try:
        return _TERMINATION_FUNCTIONS[robot]
    except KeyError as error:
        raise ValueError(
            f"No termination rule is ported for {robot!r}. Ported are "
            f"{sorted(_TERMINATION_FUNCTIONS)}."
        ) from error
