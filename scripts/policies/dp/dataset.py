import numpy as np
import torch

from scripts.shared.data import EpisodeStore
from scripts.shared.normalization import MinMaxNormalizer


class Dataset(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        *,
        episode_store: EpisodeStore,
        prediction_horizon_length: int,
        observation_horizon_length: int,
        action_horizon_length: int,
    ) -> None:
        if (
            episode_store.intended_prediction_horizon is not None
            and episode_store.intended_prediction_horizon != prediction_horizon_length
        ):
            raise ValueError(
                f"This dataset's augmented episodes were laid out for a "
                f"prediction horizon of "
                f"{episode_store.intended_prediction_horizon}, but the policy "
                f"is configured with {prediction_horizon_length}. Training on "
                f"it would silently produce duplicated or missing windows; "
                f"regenerate the augmentation with the current horizon."
            )

        observation_batch = torch.from_numpy(
            episode_store.states
        ).float()  # (total_timestep_count, observation_dimensionality)
        action_batch = torch.from_numpy(
            episode_store.actions
        ).float()  # (total_timestep_count, action_dimensionality)
        episode_ends = torch.from_numpy(
            episode_store.episode_end_indices
        )  # (episode_count,)
        artificial_end_mask = torch.from_numpy(
            episode_store.artificial_end_mask
            if episode_store.artificial_end_mask is not None
            else np.zeros(len(episode_store.episode_end_indices), dtype=np.bool_)
        )  # (episode_count,)

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

        self._window_indices = _build_window_indices(
            episode_ends=episode_ends,
            prediction_horizon_length=prediction_horizon_length,
            observation_padding_length=observation_horizon_length - 1,
            action_padding_length=action_horizon_length - 1,
            artificial_end_mask=artificial_end_mask,
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
    prediction_horizon_length: int,
    observation_padding_length: int,
    action_padding_length: int,
    artificial_end_mask: torch.Tensor,  # (episode_count,)
) -> torch.Tensor:  # (window_count, 4) columns: source_read_start, source_read_end, output_write_start, output_write_end
    window_indices: list[list[int]] = []

    for episode_index, episode_end_position in enumerate(episode_ends):
        episode_start_position = (
            int(episode_ends[episode_index - 1]) if episode_index > 0 else 0
        )
        episode_length = int(episode_end_position) - episode_start_position

        first_window_offset = -observation_padding_length
        # An artificial episode end is a mid-task cut: windows padded past it
        # would repeat the last action as a label, teaching the policy to stop
        # mid-task. Such episodes only contribute windows whose actions are
        # fully real.
        last_window_offset = (
            episode_length - prediction_horizon_length
            if bool(artificial_end_mask[episode_index])
            else episode_length - prediction_horizon_length + action_padding_length
        )

        for window_offset in range(first_window_offset, last_window_offset + 1):
            source_read_start_position = episode_start_position + max(window_offset, 0)
            source_read_end_position = episode_start_position + min(
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

    return torch.tensor(window_indices, dtype=torch.long)  # (window_count, 4)


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
