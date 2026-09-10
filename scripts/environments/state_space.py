from dataclasses import dataclass
from typing import Protocol

import torch


class StateSpace(Protocol):
    """The geometry of one environment's state vector.

    Most of the pipeline treats a state as a plain vector. This breaks down
    where a state carries a rotation, because rotations do not add and
    subtract like numbers. Every place that adds, subtracts or averages
    states goes through this interface, which lets the rest of the pipeline
    stay free of environment knowledge.
    """

    @property
    def dimensionality(self) -> int: ...

    @property
    def network_input_dimensionality(self) -> int: ...

    @property
    def picture(self) -> "StatePicture":
        """What a diagnostic plot can show of one state."""
        ...

    def add(
        self,
        *,
        first_addend_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        second_addend_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        ...

    def subtract(
        self,
        *,
        minuend_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        subtrahend_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        ...

    def average(
        self,
        *,
        state_batch: torch.Tensor,  # (member_count, batch_size, state_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        ...

    def create_network_input(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, network_input_dimensionality)
        ...


@dataclass(frozen=True)
class PositionGroup:
    """Two state entries that name a place on the plane."""

    label: str
    entries: slice


@dataclass(frozen=True)
class StatePicture:
    """What a diagnostic plot can show of a state.

    A state is a long vector, and most of its entries mean nothing on paper.
    These are the ones that do: the places the moved things occupy, and the
    one angle, where the environment has one.
    """

    position_groups: tuple[PositionGroup, ...]
    horizontal_limits: tuple[float, float]
    vertical_limits: tuple[float, float]
    position_unit: str
    orientation_index: int | None
