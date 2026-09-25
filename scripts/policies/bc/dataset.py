from collections.abc import Iterator
from dataclasses import dataclass

import torch

from scripts.shared.normalization import MinMaxNormalizer, ZScoreNormalizer
from scripts.shared.state_action_transitions import StateActionTransitions


@dataclass(frozen=True, slots=True)
class Batch:
    states: torch.Tensor  # (batch_size, state_dimensionality)
    actions: torch.Tensor  # (batch_size, action_dimensionality)


class Dataset:
    def __init__(self, *, state_action_transitions: StateActionTransitions) -> None:
        self._states = torch.from_numpy(
            state_action_transitions.states
        ).float()  # (pair_count, state_dimensionality)

        self._actions = torch.from_numpy(
            state_action_transitions.actions
        ).float()  # (pair_count, action_dimensionality)

        self._device = self._states.device

        self.observation_normalizer = ZScoreNormalizer(value_batch=self._states)
        self.action_normalizer = MinMaxNormalizer(value_batch=self._actions)

    def __len__(self) -> int:
        return len(self._states)

    def to(self, *, device: torch.device) -> None:
        self._states = self._states.to(device=device)
        self._actions = self._actions.to(device=device)
        self._device = device

    def batches(self, *, batch_size: int) -> Iterator[Batch]:
        indices = torch.randperm(len(self), device=self._device)
        for batch_indices in indices.split(batch_size):
            yield Batch(
                states=self._states[batch_indices], actions=self._actions[batch_indices]
            )
