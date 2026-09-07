from dataclasses import dataclass

import numpy as np
import torch
from numpy import typing as npt

from scripts.shared.episode_store import EpisodeStore
from scripts.shared.normalization import MinMaxNormalizer, ZScoreNormalizer
from scripts.shared.pusht_reward import calculate_reward, calculate_termination


@dataclass(frozen=True)
class Transitions:
    states: npt.NDArray[np.float32]  # (transition_count, state_dimensionality)
    actions: npt.NDArray[np.float32]  # (transition_count, action_dimensionality)
    next_states: npt.NDArray[np.float32]  # (transition_count, state_dimensionality)
    rewards: npt.NDArray[np.float32]  # (transition_count, 1)
    terminations: npt.NDArray[np.float32]  # (transition_count, 1)


def build_transitions(
    *,
    episode_store: EpisodeStore,
    state_normalizer: ZScoreNormalizer,
    action_normalizer: MinMaxNormalizer,
) -> Transitions:
    has_successor = np.ones(shape=len(episode_store.states), dtype=np.bool_)
    has_successor[episode_store.episode_end_indices - 1] = False
    source_indices = np.flatnonzero(has_successor)

    next_states = episode_store.states[source_indices + 1]

    return Transitions(
        states=state_normalizer.normalize(
            value_batch=torch.from_numpy(episode_store.states[source_indices]).float()
        ).numpy(),
        actions=action_normalizer.normalize(
            value_batch=torch.from_numpy(episode_store.actions[source_indices]).float()
        ).numpy(),
        next_states=state_normalizer.normalize(
            value_batch=torch.from_numpy(next_states).float()
        ).numpy(),
        rewards=calculate_reward(state_batch=next_states)
        .astype(np.float32)
        .reshape(-1, 1),
        terminations=calculate_termination(state_batch=next_states)
        .astype(np.float32)
        .reshape(-1, 1),
    )
