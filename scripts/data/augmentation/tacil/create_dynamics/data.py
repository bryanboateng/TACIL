from dataclasses import dataclass

import numpy as np
from numpy import typing as npt

from scripts.shared.episodes import Episodes, calculate_episode_start_indices


@dataclass
class Transitions:
    states: npt.NDArray[np.float32]  # (transition_count, state_dimensionality)
    actions: npt.NDArray[np.float32]  # (transition_count, action_dimensionality)
    next_states: npt.NDArray[np.float32]  # (transition_count, state_dimensionality)


def extract_transitions(*, episodes: Episodes) -> list[Transitions]:

    episode_start_indices = calculate_episode_start_indices(
        episode_end_indices=episodes.episode_end_indices
    )  # (episode_count,)

    transitions_per_episode: list[Transitions] = []
    for episode_start_index, episode_end_index in zip(
        episode_start_indices, episodes.episode_end_indices, strict=True
    ):
        episode_states = episodes.states[
            episode_start_index:episode_end_index
        ]  # (transition_count, state_dimensionality)

        episode_actions = episodes.actions[
            episode_start_index:episode_end_index
        ]  # (transition_count, action_dimensionality)

        # Drop the last timestep to form (s_t, a_t, s_{t+1}) triples,
        # since the terminal state has no corresponding action.
        transitions_per_episode.append(
            Transitions(
                states=episode_states[
                    :-1
                ],  # (episode_length - 1, state_dimensionality)
                actions=episode_actions[
                    :-1
                ],  # (episode_length - 1, action_dimensionality)
                next_states=episode_states[
                    1:
                ],  # (episode_length - 1, state_dimensionality)
            )
        )

    return transitions_per_episode


def concatenate_transitions(
    *, transitions_per_episode: list[Transitions]
) -> Transitions:
    return Transitions(
        states=np.concatenate(
            [transitions.states for transitions in transitions_per_episode], axis=0
        ),
        actions=np.concatenate(
            [transitions.actions for transitions in transitions_per_episode], axis=0
        ),
        next_states=np.concatenate(
            [transitions.next_states for transitions in transitions_per_episode], axis=0
        ),
    )


def build_windows(
    *, transitions_per_episode: list[Transitions], window_length: int
) -> tuple[
    npt.NDArray[
        np.float32
    ],  # state_windows: (total_window_count, window_length + 1, state_dimensionality)
    npt.NDArray[
        np.float32
    ],  # action_windows: (total_window_count, window_length, action_dimensionality)
]:
    action_window_list: list[npt.NDArray[np.float32]] = []
    state_window_list: list[npt.NDArray[np.float32]] = []

    for transitions in transitions_per_episode:
        episode_length = transitions.actions.shape[0]
        if episode_length < window_length:
            raise ValueError(
                f"Episode of length {episode_length} is shorter than "
                f"window length {window_length}."
            )

        final_next_state = transitions.next_states[[-1]]  # (1, state_dimensionality)
        full_state_sequence = np.concatenate(
            [transitions.states, final_next_state], axis=0
        )  # (episode_length + 1, state_dimensionality)

        # One more state than actions in each window: action k is executed at
        # state k, so the window's final state has no corresponding action.
        state_windows = _build_sliding_windows(
            array=full_state_sequence, window_length=window_length + 1
        )  # (episode_window_count, window_length + 1, state_dimensionality)

        action_windows = _build_sliding_windows(
            array=transitions.actions, window_length=window_length
        )  # (episode_window_count, window_length, action_dimensionality)

        assert state_windows.shape[0] == action_windows.shape[0]

        state_window_list.append(state_windows)
        action_window_list.append(action_windows)

    return (
        np.ascontiguousarray(
            a=np.concatenate(state_window_list, axis=0)
        ),  # (total_window_count, window_length + 1, state_dimensionality)
        np.ascontiguousarray(
            a=np.concatenate(action_window_list, axis=0)
        ),  # (total_window_count, window_length, action_dimensionality)
    )


# ---- Private --------------------------------------------


def _build_sliding_windows(
    *,
    array: npt.NDArray[np.float32],  # (array_length, sample_dimensionality)
    window_length: int,
) -> npt.NDArray[np.float32]:  # (window_count, window_length, sample_dimensionality)

    sliding_windows = np.lib.stride_tricks.sliding_window_view(
        x=array, window_shape=window_length, axis=0
    )  # (window_count, sample_dimensionality, window_length)

    return np.swapaxes(
        a=sliding_windows, axis1=1, axis2=2
    )  # (window_count, window_length, sample_dimensionality)
