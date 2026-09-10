import torch

from scripts.environments.state_picture import StatePicture


class VectorStateSpace:
    """A state whose entries are ordinary numbers, so they add and subtract
    the way numbers do."""

    def __init__(
        self, *, dimensionality: int, action_dimensionality: int, picture: StatePicture
    ):
        self._dimensionality = dimensionality
        self._action_dimensionality = action_dimensionality
        self._picture = picture

    @property
    def dimensionality(self) -> int:
        return self._dimensionality

    @property
    def network_input_dimensionality(self) -> int:
        return self._dimensionality + self._action_dimensionality

    @property
    def picture(self) -> StatePicture:
        return self._picture

    def add(
        self,
        *,
        first_addend_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        second_addend_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        return first_addend_batch + second_addend_batch

    def subtract(
        self,
        *,
        minuend_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        subtrahend_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        return minuend_batch - subtrahend_batch

    def average(
        self,
        *,
        state_batch: torch.Tensor,  # (member_count, batch_size, state_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        return state_batch.mean(dim=0)

    def create_network_input(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, network_input_dimensionality)
        return torch.cat(tensors=[state_batch, action_batch], dim=-1)
