from typing import Protocol

import torch

from scripts.environments.state_picture import StatePicture


class StateSpace(Protocol):
    @property
    def dimensionality(self) -> int: ...

    @property
    def action_dimensionality(self) -> int: ...

    @property
    def network_input_dimensionality(self) -> int: ...

    @property
    def picture(self) -> StatePicture: ...

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
