from dataclasses import dataclass
from logging import Logger

import numpy as np
from numpy import typing as npt
from wandb import Artifact

from scripts.shared.data import load_demonstration_data_from_artifact


@dataclass
class Transitions:
    states: npt.NDArray[np.float32]  #  (episode_length, state_dimensionality)
    actions: npt.NDArray[np.float32]  # (episode_length, action_dimensionality)
    next_states: npt.NDArray[np.float32]  # (episode_length, state_dimensionality)


class NoValidWindowsError(ValueError):
    pass


def load_episodes(*, artifact: Artifact, logger: Logger) -> list[Transitions]:

    demonstration_data = load_demonstration_data_from_artifact(
        artifact=artifact, logger=logger
    )

    episode_start_indices = np.concatenate(
        [[0], demonstration_data.episode_end_indices[:-1]]
    )  # (episode_count,)

    episodes: list[Transitions] = []
    for episode_start_index, episode_end_index in zip(
        episode_start_indices, demonstration_data.episode_end_indices
    ):
        episode_states = demonstration_data.states[
            episode_start_index:episode_end_index
        ]  # (episode_length, state_dimensionality)

        episode_actions = demonstration_data.actions[
            episode_start_index:episode_end_index
        ]  # (episode_length, action_dimensionality)

        # Drop the last timestep to form (s_t, a_t, s_{t+1}) triples,
        # since the terminal state has no corresponding action.
        episodes.append(
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

    return episodes


def concatenate_episodes(*, episodes: list[Transitions]) -> Transitions:
    return Transitions(
        states=np.concatenate([episode.states for episode in episodes], axis=0),
        actions=np.concatenate([episode.actions for episode in episodes], axis=0),
        next_states=np.concatenate(
            [episode.next_states for episode in episodes], axis=0
        ),
    )


def build_windows(
    *, episodes: list[Transitions], window_length: int
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

    for episode in episodes:
        final_next_state = episode.next_states[[-1]]  # (1, state_dimensionality)
        full_state_sequence = np.concatenate(
            [episode.states, final_next_state], axis=0
        )  # (episode_length + 1, state_dimensionality)

        if episode.actions.shape[0] < window_length:
            continue

        state_windows = _build_sliding_windows(
            array=full_state_sequence, window_length=window_length + 1
        )  # (episode_window_count, window_length + 1, state_dimensionality)

        action_windows = _build_sliding_windows(
            array=episode.actions, window_length=window_length
        )  # (episode_window_count, window_length, action_dimensionality)

        assert state_windows.shape[0] == action_windows.shape[0]

        state_window_list.append(state_windows)
        action_window_list.append(action_windows)

    if not action_window_list:
        raise NoValidWindowsError(
            f"No episodes long enough to build windows of horizon {window_length}."
        )

    return (
        np.ascontiguousarray(
            a=np.concatenate(state_window_list, axis=0)
        ),  # (total_window_count, window_length + 1, state_dimensionality)
        np.ascontiguousarray(
            a=np.concatenate(action_window_list, axis=0)
        ),  # (total_window_count, window_length, action_dimensionality)
    )


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
