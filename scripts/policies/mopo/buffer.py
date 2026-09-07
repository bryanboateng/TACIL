"""Replay buffers of MOPO.

Ported from `~/uni/masterarbeit/OfflineRL-Kit`, `offlinerlkit/buffer/buffer.py`.
One buffer holds the real transitions, the other the imagined ones.
"""

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

        self._states = np.zeros(
            shape=(capacity, state_dimensionality), dtype=np.float32
        )
        self._actions = np.zeros(
            shape=(capacity, action_dimensionality), dtype=np.float32
        )
        self._next_states = np.zeros(
            shape=(capacity, state_dimensionality), dtype=np.float32
        )
        self._rewards = np.zeros(shape=(capacity, 1), dtype=np.float32)
        self._terminations = np.zeros(shape=(capacity, 1), dtype=np.float32)

        self._write_index = 0
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    def add(self, *, transitions: Transitions) -> None:
        count = len(transitions.states)
        indices = np.arange(self._write_index, self._write_index + count) % (
            self._capacity
        )

        self._states[indices] = transitions.states
        self._actions[indices] = transitions.actions
        self._next_states[indices] = transitions.next_states
        self._rewards[indices] = transitions.rewards
        self._terminations[indices] = transitions.terminations

        self._write_index = (self._write_index + count) % self._capacity
        self._size = min(self._size + count, self._capacity)

    def sample(self, *, count: int, generator: np.random.Generator) -> Batch:
        indices = generator.integers(low=0, high=self._size, size=count)
        return Batch(
            states=self._to_tensor(value_batch=self._states[indices]),
            actions=self._to_tensor(value_batch=self._actions[indices]),
            next_states=self._to_tensor(value_batch=self._next_states[indices]),
            rewards=self._to_tensor(value_batch=self._rewards[indices]),
            terminations=self._to_tensor(value_batch=self._terminations[indices]),
        )

    def sample_states(
        self, *, count: int, generator: np.random.Generator
    ) -> npt.NDArray[np.float32]:  # (count, state_dimensionality)
        return self._states[generator.integers(low=0, high=self._size, size=count)]

    # ---- Private --------------------------------------------

    def _to_tensor(self, *, value_batch: npt.NDArray[np.float32]) -> torch.Tensor:
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
