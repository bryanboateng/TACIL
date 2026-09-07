"""Transitions for MOPO, built from expert episodes.

The demonstrations were recorded for imitation learning and carry no reward.
MOPO needs one per step, so the reward is computed afterwards with the reward
function of the environment. This restores what a reinforcement learning
dataset such as D4RL records while it is collected.

Actions are normalized to [-1, 1]. The reference implementation assumes an
action space that is symmetric around zero, which the PushT action space
(0 to 512 in each dimension) is not.

States are normalized as well, the way `run_example/run_rambo.py` does it for
the other model-based algorithm of the reference. PushT states reach 512, and a
Soft Actor-Critic learns badly from numbers that large. Reward and termination
are read off the raw states before the states are normalized.
"""

from dataclasses import dataclass

import numpy as np
import torch
from numpy import typing as npt

from scripts.shared.episode_store import EpisodeStore
from scripts.shared.normalization import MinMaxNormalizer, ZScoreNormalizer
from scripts.shared.pusht_reward import (
    calculate_reward_batch,
    calculate_termination_batch,
)


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
    states = episode_store.states
    actions = (
        action_normalizer.normalize(
            value_batch=torch.from_numpy(episode_store.actions).float()
        )
        .numpy()
        .astype(np.float32)
    )

    # The last step of an episode has no successor.
    is_start_of_transition = np.ones(shape=len(states), dtype=np.bool_)
    is_start_of_transition[episode_store.episode_end_indices - 1] = False
    start_indices = np.flatnonzero(is_start_of_transition)

    next_states = states[start_indices + 1]

    return Transitions(
        states=_normalize_states(
            state_batch=states[start_indices], state_normalizer=state_normalizer
        ),
        actions=actions[start_indices],
        next_states=_normalize_states(
            state_batch=next_states, state_normalizer=state_normalizer
        ),
        rewards=calculate_reward_batch(state_batch=next_states)
        .astype(np.float32)
        .reshape(-1, 1),
        terminations=calculate_termination_batch(state_batch=next_states)
        .astype(np.float32)
        .reshape(-1, 1),
    )


# ---- Private --------------------------------------------


def _normalize_states(
    *,
    state_batch: npt.NDArray[np.float32],  # (transition_count, state_dimensionality)
    state_normalizer: ZScoreNormalizer,
) -> npt.NDArray[np.float32]:  # (transition_count, state_dimensionality)
    return (
        state_normalizer.normalize(value_batch=torch.from_numpy(state_batch).float())
        .numpy()
        .astype(np.float32)
    )
