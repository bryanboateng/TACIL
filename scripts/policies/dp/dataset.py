from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch

from scripts.policies.dp.config import HorizonLengthConfig
from scripts.shared.episodes import Episodes, calculate_episode_lengths
from scripts.shared.normalization import MinMaxNormalizer


@dataclass(frozen=True, slots=True)
class WindowBatch:
    observation_windows: (
        torch.Tensor
    )  # (window_count, observation_horizon_length, observation_dimensionality)
    action_windows: (
        torch.Tensor
    )  # (window_count, prediction_horizon_length, action_dimensionality)


class Dataset:
    def __init__(
        self,
        *,
        expert_episodes: Episodes,
        augmented_episodes: Episodes | None,
        horizon_lengths: HorizonLengthConfig,
    ) -> None:
        self.observation_normalizer, self.action_normalizer = _create_normalizers(
            all_episodes=[expert_episodes]
            + ([augmented_episodes] if augmented_episodes is not None else [])
        )

        expert_windows = _create_windows(
            episodes=expert_episodes,
            observation_normalizer=self.observation_normalizer,
            action_normalizer=self.action_normalizer,
            window_length=horizon_lengths.prediction,
            observation_horizon_length=horizon_lengths.observation,
            lead_in_length=horizon_lengths.observation - 1,
            run_out_length=horizon_lengths.execution - 1,
        )

        if augmented_episodes is None:
            windows = expert_windows
        else:
            episode_lengths = calculate_episode_lengths(
                episode_end_indices=augmented_episodes.episode_end_indices
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
                observation_normalizer=self.observation_normalizer,
                action_normalizer=self.action_normalizer,
                window_length=horizon_lengths.prediction,
                observation_horizon_length=horizon_lengths.observation,
                lead_in_length=0,
                run_out_length=0,
            )
            windows = WindowBatch(
                observation_windows=torch.cat(
                    [
                        expert_windows.observation_windows,
                        augmented_windows.observation_windows,
                    ]
                ),
                action_windows=torch.cat(
                    [expert_windows.action_windows, augmented_windows.action_windows]
                ),
            )

        self._windows = windows
        self._device = self._windows.action_windows.device

    def __len__(self) -> int:
        return len(self._windows.action_windows)

    def to(self, *, device: torch.device) -> None:
        self._windows = WindowBatch(
            observation_windows=self._windows.observation_windows.to(device=device),
            action_windows=self._windows.action_windows.to(device=device),
        )
        self._device = device

    def batches(self, *, batch_size: int) -> Iterator[WindowBatch]:
        indices = torch.randperm(len(self), device=self._device)
        for batch_indices in indices.split(batch_size):
            yield WindowBatch(
                observation_windows=self._windows.observation_windows[batch_indices],
                action_windows=self._windows.action_windows[batch_indices],
            )


# ---- Private --------------------------------------------


def _create_normalizers(
    *, all_episodes: list[Episodes]
) -> tuple[MinMaxNormalizer, MinMaxNormalizer]:
    return (
        MinMaxNormalizer(
            value_batch=torch.from_numpy(
                np.concatenate([episodes.states for episodes in all_episodes], axis=0)
            ).float()
        ),
        MinMaxNormalizer(
            value_batch=torch.from_numpy(
                np.concatenate([episodes.actions for episodes in all_episodes], axis=0)
            ).float()
        ),
    )


def _create_windows(
    *,
    episodes: Episodes,
    observation_normalizer: MinMaxNormalizer,
    action_normalizer: MinMaxNormalizer,
    window_length: int,
    observation_horizon_length: int,
    lead_in_length: int,
    run_out_length: int,
) -> WindowBatch:
    window_timesteps = _get_window_timesteps(
        episode_end_indices=episodes.episode_end_indices,
        window_length=window_length,
        lead_in_length=lead_in_length,
        run_out_length=run_out_length,
    )  # (window_count, window_length)

    normalized_observations = observation_normalizer.normalize(
        value_batch=torch.from_numpy(episodes.states).float()
    )  # (timestep_count, observation_dimensionality)
    normalized_actions = action_normalizer.normalize(
        value_batch=torch.from_numpy(episodes.actions).float()
    )  # (timestep_count, action_dimensionality)

    # A window is as long as the prediction, and only its first
    # `observation_horizon_length` steps become the observation history:

    #     step         0 1 2 3 4 5 6 7 8 9 A B C D E F
    #     observation  o o
    #     prediction   a a a a a a a a a a a a a a a a
    return WindowBatch(
        observation_windows=normalized_observations[
            window_timesteps[:, :observation_horizon_length]
        ],
        action_windows=normalized_actions[window_timesteps],
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
) -> torch.Tensor:  # (window_count, window_length)

    # A window may start before the episode and may end after it.
    earliest_start = -lead_in_length
    latest_start = episode_length - window_length + run_out_length

    window_count = max(latest_start + 1 - earliest_start, 0)

    start_timesteps = earliest_start + torch.arange(end=window_count)  # (window_count,)

    # From its start, a window covers `window_length` timesteps in a row.
    covered_timesteps = (
        start_timesteps[:, None] + torch.arange(end=window_length)[None, :]
    )  # (window_count, window_length)

    # Reading before or after the episode would land in a neighboring one.
    return covered_timesteps.clamp(min=0, max=episode_length - 1)


def _calculate_episode_spans(
    *,
    episode_end_indices: npt.NDArray[np.int64],  # (episode_count,)
) -> list[tuple[int, int]]:
    ends = [int(episode_end_index) for episode_end_index in episode_end_indices]
    starts = [0] + ends[:-1]
    return list(zip(starts, ends, strict=True))
