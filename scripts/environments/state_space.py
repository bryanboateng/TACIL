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
