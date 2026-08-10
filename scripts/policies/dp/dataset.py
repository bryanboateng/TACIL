from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch

from scripts.policies.dp.config import HorizonLengthConfig
from scripts.shared.f import EpisodeStore
from scripts.shared.normalization import MinMaxNormalizer


@dataclass(frozen=True, slots=True)
class _Normalizers:
    observation: MinMaxNormalizer
    action: MinMaxNormalizer


@dataclass(frozen=True, slots=True)
class _Windows:
    observations: (
        torch.Tensor
    )  # (window_count, observation_horizon_length, observation_dimensionality)
    actions: (
        torch.Tensor
    )  # (window_count, prediction_horizon_length, action_dimensionality)


class Dataset(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        *,
        expert_episodes: EpisodeStore,
        augmented_episodes: EpisodeStore | None,
        horizon_lengths: HorizonLengthConfig,
    ) -> None:
        normalizers = _fit_normalizers(
            episode_stores=[expert_episodes]
            + ([augmented_episodes] if augmented_episodes is not None else [])
        )
        self.observation_normalizer = normalizers.observation
        self.action_normalizer = normalizers.action

        expert_windows = _create_windows(
            episodes=expert_episodes,
            normalizers=normalizers,
            window_length=horizon_lengths.prediction,
            observation_horizon_length=horizon_lengths.observation,
            # The policy repeats its first observation until it has a full
            # history, so a window may begin before the episode does.
            lead_in_length=horizon_lengths.observation - 1,
            # The policy plans once more near the end and then runs a whole
            # execution horizon, so a window may reach past the episode.
            run_out_length=horizon_lengths.execution - 1,
        )

        if augmented_episodes is None:
            self._windows = expert_windows
        else:
            episode_lengths = np.diff(
                a=augmented_episodes.episode_end_indices, prepend=0
            )  # (episode_count,)
            longest_episode_length = int(episode_lengths.max())
            if longest_episode_length < horizon_lengths.prediction:
                raise ValueError(
                    f"None of the {len(episode_lengths)} augmented episodes is long enough "
                    f"for one window: the longest is {longest_episode_length} timesteps, "
                    f"the prediction horizon is {horizon_lengths.prediction}."
                )
            augmented_windows = _create_windows(
                episodes=augmented_episodes,
                normalizers=normalizers,
                window_length=horizon_lengths.prediction,
                observation_horizon_length=horizon_lengths.observation,
                lead_in_length=0,
                run_out_length=0,
            )
            self._windows = _Windows(
                observations=torch.cat(
                    [expert_windows.observations, augmented_windows.observations]
                ),
                actions=torch.cat([expert_windows.actions, augmented_windows.actions]),
            )

    def __len__(self) -> int:
        return len(self._windows.actions)

    def __getitem__(
        self, index: int
    ) -> tuple[
        torch.Tensor, torch.Tensor
    ]:  # (observation_horizon_length, observation_dimensionality), (prediction_horizon_length, action_dimensionality)
        return self._windows.observations[index], self._windows.actions[index]


def _fit_normalizers(*, episode_stores: list[EpisodeStore]) -> _Normalizers:
    return _Normalizers(
        observation=MinMaxNormalizer.from_values(
            value_batch=torch.from_numpy(
                np.concatenate(
                    [episode_store.states for episode_store in episode_stores], axis=0
                )
            ).float()
        ),
        action=MinMaxNormalizer.from_values(
            value_batch=torch.from_numpy(
                np.concatenate(
                    [episode_store.actions for episode_store in episode_stores], axis=0
                )
            ).float()
        ),
    )


def _create_windows(
    *,
    episodes: EpisodeStore,
    normalizers: _Normalizers,
    window_length: int,
    observation_horizon_length: int,
    lead_in_length: int,
    run_out_length: int,
) -> _Windows:
    window_timesteps = _get_window_timesteps(
        episode_end_indices=episodes.episode_end_indices,
        window_length=window_length,
        lead_in_length=lead_in_length,
        run_out_length=run_out_length,
    )  # (window_count, window_length)

    normalized_observations = normalizers.observation.normalize(
        value_batch=torch.from_numpy(episodes.states).float()
    )  # (timestep_count, observation_dimensionality)
    normalized_actions = normalizers.action.normalize(
        value_batch=torch.from_numpy(episodes.actions).float()
    )  # (timestep_count, action_dimensionality)

    # A window is as long as the prediction, and only its first
    # `observation_horizon_length` steps become the observation history:

    #     step         0 1 2 3 4 5 6 7 8 9 A B C D E F
    #     observation  o o
    #     prediction   a a a a a a a a a a a a a a a a
    return _Windows(
        observations=normalized_observations[
            window_timesteps[:, :observation_horizon_length]
        ],
        actions=normalized_actions[window_timesteps],
    )


def _get_window_timesteps(
    *,
    episode_end_indices: npt.NDArray[np.int64],  # (episode_count,)
    window_length: int,
    lead_in_length: int,
    run_out_length: int,
) -> torch.Tensor:  # (window_count, window_length)
    return torch.cat(
        [
            episode_start
            + _get_window_timesteps_of_one_episode(
                episode_length=episode_end - episode_start,
                window_length=window_length,
                lead_in_length=lead_in_length,
                run_out_length=run_out_length,
            )
            for episode_start, episode_end in _calculate_episode_spans(
                episode_end_indices=episode_end_indices
            )
        ]
    )


def _get_window_timesteps_of_one_episode(
    *, episode_length: int, window_length: int, lead_in_length: int, run_out_length: int
) -> (
    torch.Tensor
):  # (window_count, window_length), counted from the episode's own start

    # A window may start before the episode and may end after it.
    earliest_start = -lead_in_length
    latest_start = episode_length - window_length + run_out_length

    # An episode too short for even one window contributes none.
    window_count = max(latest_start + 1 - earliest_start, 0)

    start_timesteps = earliest_start + torch.arange(end=window_count)  # (window_count,)

    # From its start, a window covers `window_length` timesteps in a row.
    covered_timesteps = (
        start_timesteps[:, None] + torch.arange(end=window_length)[None, :]
    )  # (window_count, window_length)

    # Reading before or after the episode would land in a neighboring one.
    # Pulling those timesteps back to the edge repeats the edge value instead.
    return covered_timesteps.clamp(min=0, max=episode_length - 1)


def _calculate_episode_spans(
    *,
    episode_end_indices: npt.NDArray[np.int64],  # (episode_count,)
) -> list[tuple[int, int]]:
    ends = [int(episode_end_index) for episode_end_index in episode_end_indices]
    starts = [0] + ends[:-1]
    return list(zip(starts, ends, strict=True))
