from dataclasses import dataclass

import numpy as np
import torch
from numpy import typing as npt

from scripts.policies.mopo.transitions import Transitions


@dataclass(frozen=True)
class Batch:
    states: torch.Tensor  # (batch_size, state_dimensionality)
    actions: torch.Tensor  # (batch_size, action_dimensionality)
    next_states: torch.Tensor  # (batch_size, state_dimensionality)
    rewards: torch.Tensor  # (batch_size, 1)
    terminations: torch.Tensor  # (batch_size, 1)


class ReplayBuffer:
    def __init__(
        self,
        *,
        capacity: int,
        state_dimensionality: int,
        action_dimensionality: int,
        device: torch.device,
    ) -> None:
        self._capacity = capacity
        self._device = device

        self._states = torch.zeros(capacity, state_dimensionality, device=device)
        self._actions = torch.zeros(capacity, action_dimensionality, device=device)
        self._next_states = torch.zeros(capacity, state_dimensionality, device=device)
        self._rewards = torch.zeros(capacity, 1, device=device)
        self._terminations = torch.zeros(capacity, 1, device=device)

        self._write_index = 0
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    def add(self, *, transitions: Transitions) -> None:
        count = len(transitions.states)
        indices = self._to_tensor(
            value_batch=np.arange(self._write_index, self._write_index + count)
            % self._capacity
        )

        self._states[indices] = self._to_tensor(value_batch=transitions.states)
        self._actions[indices] = self._to_tensor(value_batch=transitions.actions)
        self._next_states[indices] = self._to_tensor(
            value_batch=transitions.next_states
        )
        self._rewards[indices] = self._to_tensor(value_batch=transitions.rewards)
        self._terminations[indices] = self._to_tensor(
            value_batch=transitions.terminations
        )

        self._write_index = (self._write_index + count) % self._capacity
        self._size = min(self._size + count, self._capacity)

    def sample(self, *, count: int) -> Batch:
        indices = self._draw_indices(count=count)
        return Batch(
            states=self._states[indices],
            actions=self._actions[indices],
            next_states=self._next_states[indices],
            rewards=self._rewards[indices],
            terminations=self._terminations[indices],
        )

    def sample_states(
        self, *, count: int
    ) -> npt.NDArray[np.float32]:  # (count, state_dimensionality)
        return self._states[self._draw_indices(count=count)].cpu().numpy()

    # ---- Private --------------------------------------------

    def _draw_indices(self, *, count: int) -> torch.Tensor:  # (count,)
        return torch.randint(low=0, high=self._size, size=(count,), device=self._device)

    def _to_tensor(self, *, value_batch: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(data=value_batch, device=self._device)


def concatenate_batches(*, first: Batch, second: Batch) -> Batch:
    return Batch(
        states=torch.cat(tensors=[first.states, second.states], dim=0),
        actions=torch.cat(tensors=[first.actions, second.actions], dim=0),
        next_states=torch.cat(tensors=[first.next_states, second.next_states], dim=0),
        rewards=torch.cat(tensors=[first.rewards, second.rewards], dim=0),
        terminations=torch.cat(
            tensors=[first.terminations, second.terminations], dim=0
        ),
    )
