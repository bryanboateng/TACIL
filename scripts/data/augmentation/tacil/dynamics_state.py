"""The dynamics state can carry history the policy never sees.

What the numbers inside a state mean lives in scripts/shared/pusht_state.py;
this module only stacks and unstacks frames, which works for any state.
"""

import numpy as np

from scripts.shared.data import EpisodeStore


def stack_previous_state(*, episode_store: EpisodeStore) -> EpisodeStore:
    """Concatenate each state with its predecessor inside the episode.

    The first timestep of an episode has no predecessor; it reuses itself,
    which keeps episode lengths and action alignment unchanged.
    """
    episode_start_indices = np.concatenate(
        [[0], episode_store.episode_end_indices[:-1]]
    )

    previous_states = np.roll(episode_store.states, shift=1, axis=0)
    # Episode starts must not look into the previous episode.
    previous_states[episode_start_indices] = episode_store.states[
        episode_start_indices
    ]

    return EpisodeStore(
        states=np.concatenate([episode_store.states, previous_states], axis=1),
        actions=episode_store.actions,
        episode_end_indices=episode_store.episode_end_indices,
        artificial_end_mask=episode_store.artificial_end_mask,
        intended_prediction_horizon=episode_store.intended_prediction_horizon,
    )


def current_state_part(
    *,
    stacked_states: np.ndarray,  # (..., state_dimensionality * 2)
) -> np.ndarray:  # (..., state_dimensionality)
    """The current-frame half of a stacked state — what the policy sees."""
    return stacked_states[..., : stacked_states.shape[-1] // 2]
