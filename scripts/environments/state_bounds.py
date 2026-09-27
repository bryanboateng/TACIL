from dataclasses import dataclass

import numpy as np
from numpy import typing as npt


@dataclass(frozen=True)
class Bound:
    entries: slice
    minimum: float
    maximum: float


def contains(
    *,
    state_batch: npt.NDArray[np.float32],  # (batch_size, state_dimensionality)
    bounds: tuple[Bound, ...],
) -> npt.NDArray[np.bool_]:  # (batch_size,)
    inside_batch = np.isfinite(state_batch).all(axis=-1)  # (batch_size,)
    for bound in bounds:
        entry_batch = state_batch[:, bound.entries]
        inside_batch = (
            inside_batch
            & (entry_batch > bound.minimum).all(axis=-1)
            & (entry_batch < bound.maximum).all(axis=-1)
        )
    return np.asarray(a=inside_batch)
