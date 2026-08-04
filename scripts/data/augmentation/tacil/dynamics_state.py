"""How the dynamics state differs from the state the policy sees.

Both transforms are applied to the data rather than inside the model, so
training, the ensemble signals, the Newton solver and the diagnostics keep
working on plain vectors:

- The block angle is recorded wrapped into [0, 2π), so a full turn shows up as
  a jump of −2π. That jump is bookkeeping, not motion — unwrap_angles removes
  it, wrap_angles puts labels back into the range the policy uses.
- With include_previous_state the state additionally carries s_{t-1}, so the
  model can read velocities off the difference. Labels are cut back to the
  current frame with current_state_part.
"""

import numpy as np

from scripts.shared.data import EpisodeStore

# PushT state layout: agent x/y, block x/y, block angle.
_ANGLE_INDEX = 4
_FULL_TURN = 2.0 * np.pi


def unwrap_angles(*, episode_store: EpisodeStore) -> EpisodeStore:
    """Undo the [0, 2π) wrapping of the block angle inside every episode.

    A wrap is a −2π step that no model can predict: it happens wherever the
    block passes one fixed orientation, and it dwarfs the real rotation per
    step — in the PushT data it inflates the typical angular step by 12x.
    """
    states = episode_store.states.copy()
    episode_start_indices = np.concatenate(
        [[0], episode_store.episode_end_indices[:-1]]
    )

    for start_index, end_index in zip(
        episode_start_indices, episode_store.episode_end_indices
    ):
        states[start_index:end_index, _ANGLE_INDEX] = np.unwrap(
            p=states[start_index:end_index, _ANGLE_INDEX]
        )

    return EpisodeStore(
        states=states,
        actions=episode_store.actions,
        episode_end_indices=episode_store.episode_end_indices,
        artificial_end_mask=episode_store.artificial_end_mask,
        intended_prediction_horizon=episode_store.intended_prediction_horizon,
    )


def wrap_angles(
    *,
    states: np.ndarray,  # (..., state_dimensionality)
) -> np.ndarray:  # (..., state_dimensionality)
    """Put the block angle back into [0, 2π), the range the policy and the
    environment use."""
    wrapped_states = states.copy()
    wrapped_states[..., _ANGLE_INDEX] %= _FULL_TURN
    return wrapped_states


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
