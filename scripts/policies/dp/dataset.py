import numpy as np
import torch

from scripts.shared.f import EpisodeStore
from scripts.shared.normalization import MinMaxNormalizer


class Dataset(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Windows over recorded episodes, plus windows over augmented ones.

    The two collections stay apart because they pad differently. A recorded
    episode starts at rest and ends with the task complete, so a window may run
    past either edge and repeat the edge value. An augmented episode is cut out
    of the middle of a task at both ends: padding it would invent a history it
    never had and a label that tells the policy to stop.
    """

    def __init__(
        self,
        *,
        expert_episodes: EpisodeStore,
        augmented_episodes: EpisodeStore | None,
        prediction_horizon_length: int,
        observation_horizon_length: int,
        action_horizon_length: int,
    ) -> None:
        observation_batch = torch.from_numpy(
            np.concatenate(
                [expert_episodes.states]
                + (
                    [augmented_episodes.states]
                    if augmented_episodes is not None
                    else []
                ),
                axis=0,
            )
        ).float()  # (total_timestep_count, observation_dimensionality)
        action_batch = torch.from_numpy(
            np.concatenate(
                [expert_episodes.actions]
                + (
                    [augmented_episodes.actions]
                    if augmented_episodes is not None
                    else []
                ),
                axis=0,
            )
        ).float()  # (total_timestep_count, action_dimensionality)

        self.observation_normalizer = MinMaxNormalizer.from_values(
            value_batch=observation_batch
        )
        self.action_normalizer = MinMaxNormalizer.from_values(value_batch=action_batch)

        self._normalized_observations = self.observation_normalizer.normalize(
            value_batch=observation_batch
        )  # (total_timestep_count, observation_dimensionality)
        self._normalized_actions = self.action_normalizer.normalize(
            value_batch=action_batch
        )  # (total_timestep_count, action_dimensionality)

        expert_window_indices = _build_window_indices(
            episode_ends=torch.from_numpy(expert_episodes.episode_end_indices),
            timestep_offset=0,
            prediction_horizon_length=prediction_horizon_length,
            observation_padding_length=observation_horizon_length - 1,
            action_padding_length=action_horizon_length - 1,
        )  # (window_count, 4)

        if augmented_episodes is None:
            self._window_indices = expert_window_indices
        else:
            augmented_window_indices = _build_window_indices(
                episode_ends=torch.from_numpy(augmented_episodes.episode_end_indices),
                timestep_offset=expert_episodes.states.shape[0],
                prediction_horizon_length=prediction_horizon_length,
                observation_padding_length=0,
                action_padding_length=0,
            )  # (window_count, 4)

            if len(augmented_window_indices) == 0:
                episode_lengths = np.diff(
                    augmented_episodes.episode_end_indices, prepend=0
                )
                raise ValueError(
                    f"None of the {len(episode_lengths)} augmented episodes is long "
                    f"enough for one window: the longest is "
                    f"{int(episode_lengths.max())} timesteps, the prediction horizon "
                    f"is {prediction_horizon_length}. Training would silently use the "
                    f"expert data alone. This augmentation was laid out for a policy "
                    f"that trains on single (state, action) pairs."
                )

            self._window_indices = torch.cat(
                [expert_window_indices, augmented_window_indices], dim=0
            )  # (window_count, 4)

        self._prediction_horizon_length = prediction_horizon_length
        self._observation_horizon_length = observation_horizon_length

    def __len__(self) -> int:
        return len(self._window_indices)

    def __getitem__(
        self, index: int
    ) -> tuple[
        torch.Tensor, torch.Tensor
    ]:  # (observation_horizon_length, observation_dimensionality), (prediction_horizon_length, action_dimensionality)
        (
            source_read_start_position,
            source_read_end_position,
            output_write_start_position,
            output_write_end_position,
        ) = (int(value) for value in self._window_indices[index])

        observations = _extract_window_with_edge_padding(
            source_tensor=self._normalized_observations,
            output_length=self._prediction_horizon_length,
            source_read_start_position=source_read_start_position,
            source_read_end_position=source_read_end_position,
            output_write_start_position=output_write_start_position,
            output_write_end_position=output_write_end_position,
        )  # (prediction_horizon_length, observation_dimensionality)
        actions = _extract_window_with_edge_padding(
            source_tensor=self._normalized_actions,
            output_length=self._prediction_horizon_length,
            source_read_start_position=source_read_start_position,
            source_read_end_position=source_read_end_position,
            output_write_start_position=output_write_start_position,
            output_write_end_position=output_write_end_position,
        )  # (prediction_horizon_length, action_dimensionality)

        observations = observations[
            : self._observation_horizon_length
        ]  # (observation_horizon_length, observation_dimensionality)

        return observations, actions


def _build_window_indices(
    *,
    episode_ends: torch.Tensor,  # (episode_count,)
    timestep_offset: int,
    prediction_horizon_length: int,
    observation_padding_length: int,
    action_padding_length: int,
) -> torch.Tensor:  # (window_count, 4) columns: source_read_start, source_read_end, output_write_start, output_write_end
    """Window positions for one collection of episodes.

    *timestep_offset* is where this collection starts in the concatenated
    arrays. Both padding lengths are 0 for a collection whose episodes are cut
    out of the middle of a task: it then yields only windows that lie entirely
    inside one episode.
    """
    window_indices: list[list[int]] = []

    for episode_index, episode_end_position in enumerate(episode_ends):
        episode_start_position = (
            int(episode_ends[episode_index - 1]) if episode_index > 0 else 0
        )
        episode_length = int(episode_end_position) - episode_start_position
        episode_base_position = timestep_offset + episode_start_position

        first_window_offset = -observation_padding_length
        last_window_offset = (
            episode_length - prediction_horizon_length + action_padding_length
        )

        for window_offset in range(first_window_offset, last_window_offset + 1):
            source_read_start_position = episode_base_position + max(window_offset, 0)
            source_read_end_position = episode_base_position + min(
                window_offset + prediction_horizon_length, episode_length
            )

            output_write_start_position = max(-window_offset, 0)
            output_write_end_position = min(
                episode_length - window_offset, prediction_horizon_length
            )

            window_indices.append(
                [
                    source_read_start_position,
                    source_read_end_position,
                    output_write_start_position,
                    output_write_end_position,
                ]
            )

    return torch.tensor(window_indices, dtype=torch.long).reshape(
        -1, 4
    )  # (window_count, 4)


def _extract_window_with_edge_padding(
    *,
    source_tensor: torch.Tensor,  # (total_timestep_count, feature_dimensionality)
    output_length: int,
    source_read_start_position: int,
    source_read_end_position: int,
    output_write_start_position: int,
    output_write_end_position: int,
) -> torch.Tensor:  # (output_length, feature_dimensionality)
    """Read a slice from `source_tensor` and place it inside an output window
    of length `output_length`. Any positions in the output window before
    `output_write_start_position` or after `output_write_end_position` are
    filled by repeating the first / last value of the read slice (edge
    padding).
    """
    valid_content = source_tensor[
        source_read_start_position:source_read_end_position
    ]  # (read_length, feature_dimensionality)

    if output_write_start_position == 0 and output_write_end_position == output_length:
        return valid_content

    output_window = torch.empty(
        (output_length,) + source_tensor.shape[1:],
        dtype=source_tensor.dtype,
        device=source_tensor.device,
    )  # (output_length, feature_dimensionality)

    output_window[output_write_start_position:output_write_end_position] = valid_content
    output_window[:output_write_start_position] = valid_content[0]
    output_window[output_write_end_position:] = valid_content[-1]

    return output_window  # (output_length, feature_dimensionality)
