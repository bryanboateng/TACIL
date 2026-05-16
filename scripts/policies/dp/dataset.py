import torch

from scripts.shared import load_push_t_zarr


class PushTWindowDataset(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        prediction_horizon_length: int,
        observation_horizon_length: int,
        action_horizon_length: int,
    ) -> None:
        observations_np, actions_np, episode_ends_np = load_push_t_zarr()

        observations = torch.from_numpy(observations_np).float()
        actions = torch.from_numpy(actions_np).float()
        episode_ends = torch.from_numpy(episode_ends_np)

        self.observation_normalization_range = NormalizationRange.from_values(
            values=observations
        )
        self.action_normalization_range = NormalizationRange.from_values(values=actions)

        self._normalized_observations = self.observation_normalization_range.normalize(
            values=observations
        )
        self._normalized_actions = self.action_normalization_range.normalize(
            values=actions
        )

        self._window_indices = _build_window_indices(
            episode_ends=episode_ends,
            prediction_horizon_length=prediction_horizon_length,
            observation_padding_length=observation_horizon_length - 1,
            action_padding_length=action_horizon_length - 1,
        )

        self._prediction_horizon_length = prediction_horizon_length
        self._observation_horizon_length = observation_horizon_length

    def __len__(self) -> int:
        return len(self._window_indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
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
        )
        actions = _extract_window_with_edge_padding(
            source_tensor=self._normalized_actions,
            output_length=self._prediction_horizon_length,
            source_read_start_position=source_read_start_position,
            source_read_end_position=source_read_end_position,
            output_write_start_position=output_write_start_position,
            output_write_end_position=output_write_end_position,
        )

        observations = observations[: self._observation_horizon_length]

        return observations, actions


def _build_window_indices(
    episode_ends: torch.Tensor,
    prediction_horizon_length: int,
    observation_padding_length: int,
    action_padding_length: int,
) -> torch.Tensor:
    window_indices: list[list[int]] = []

    for episode_index, episode_end_position in enumerate(episode_ends):
        episode_start_position = (
            int(episode_ends[episode_index - 1]) if episode_index > 0 else 0
        )
        episode_length = int(episode_end_position) - episode_start_position

        first_window_offset = -observation_padding_length
        last_window_offset = (
            episode_length - prediction_horizon_length + action_padding_length
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

    return torch.tensor(window_indices, dtype=torch.long)


def _extract_window_with_edge_padding(
    source_tensor: torch.Tensor,
    output_length: int,
    source_read_start_position: int,
    source_read_end_position: int,
    output_write_start_position: int,
    output_write_end_position: int,
) -> torch.Tensor:
    """Read a slice from `source_tensor` and place it inside an output window
    of length `output_length`. Any positions in the output window before
    `output_write_start_position` or after `output_write_end_position` are
    filled by repeating the first / last value of the read slice (edge
    padding).
    """
    valid_content = source_tensor[source_read_start_position:source_read_end_position]

    if output_write_start_position == 0 and output_write_end_position == output_length:
        return valid_content

    output_window = torch.empty(
        (output_length,) + source_tensor.shape[1:],
        dtype=source_tensor.dtype,
        device=source_tensor.device,
    )

    output_window[output_write_start_position:output_write_end_position] = valid_content
    output_window[:output_write_start_position] = valid_content[0]
    output_window[output_write_end_position:] = valid_content[-1]

    return output_window


class NormalizationRange:
    """Symmetric min/max normalizer mapping a feature vector to ``[-1, 1]``.

    Guards against zero-range features so constant dimensions stay finite.
    """

    def __init__(self, minimum: torch.Tensor, maximum: torch.Tensor) -> None:
        self.minimum = minimum
        self.maximum = maximum

    @classmethod
    def from_values(cls, values: torch.Tensor) -> "NormalizationRange":
        flattened = values.reshape(-1, values.shape[-1])

        return cls(
            minimum=flattened.min(dim=0).values,
            maximum=flattened.max(dim=0).values,
        )

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        value_range = (self.maximum - self.minimum).clamp(min=1e-8)
        unit = (values - self.minimum) / value_range
        return unit * 2.0 - 1.0

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        unit = (values + 1.0) / 2.0
        return unit * (self.maximum - self.minimum) + self.minimum
