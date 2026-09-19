from typing import Protocol

import torch


class NetworkInputBuilder(Protocol):
    def __call__(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, network_input_dimensionality)
        ...


class NetworkInput:
    def __init__(self, *, dimensionality: int, builder: NetworkInputBuilder) -> None:
        self.dimensionality = dimensionality
        self._builder = builder

    def build(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, network_input_dimensionality)
        return self._builder(state_batch=state_batch, action_batch=action_batch)
