from typing import Protocol

import numpy as np
from numpy import typing as npt


class Policy(Protocol):
    def __call__(
        self,
        *,
        observations: dict[
            int, npt.NDArray[np.float32]
        ],  # episode index -> (observation_dimensionality,)
    ) -> dict[
        int, npt.NDArray[np.float32]
    ]: ...  # episode index -> (action_dimensionality,)


class PolicyFactory(Protocol):
    def __call__(self, *, seeds: list[int]) -> Policy: ...
