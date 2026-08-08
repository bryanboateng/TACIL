from dataclasses import dataclass

import numpy as np
from numpy import typing as npt

from scripts.data.base.episode_store import EpisodeStore


@dataclass
class TransitionBatch:
    """(state, action) pairs with no episode structure. What a policy that
    trains on single pairs sees."""

    states: npt.NDArray[np.float32]  # (transition_count, state_dimensionality)
    actions: npt.NDArray[np.float32]  # (transition_count, action_dimensionality)


def to_transitions(*, episode_store: EpisodeStore) -> TransitionBatch:
    """Drop the episode boundaries. Every timestep becomes one sample."""
    return TransitionBatch(states=episode_store.states, actions=episode_store.actions)


def concatenate_transitions(
    *, transition_batches: list[TransitionBatch]
) -> TransitionBatch:
    return TransitionBatch(
        states=np.concatenate(
            [batch.states for batch in transition_batches], axis=0
        ),  # (transition_count, state_dimensionality)
        actions=np.concatenate(
            [batch.actions for batch in transition_batches], axis=0
        ),  # (transition_count, action_dimensionality)
    )
