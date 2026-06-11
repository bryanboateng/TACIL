from typing import Protocol

import numpy as np
from numpy import typing as npt


class Policy(Protocol):
    def __call__(
        self, observation: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]: ...


class PolicyFactory(Protocol):
    def __call__(self, seed: int) -> Policy: ...
