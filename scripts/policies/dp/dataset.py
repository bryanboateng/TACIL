from typing import Dict, List

import numpy as np
from torch.utils.data import Dataset

from scripts.shared import load_push_t_zarr


class PushTWindowDataset(Dataset):
    """Sliding windows of length ``pred_horizon`` with edge padding.

    ``obs`` is truncated to ``obs_horizon`` timesteps; ``action`` keeps the full
    prediction horizon. Both are normalised to ``[-1, 1]``.
    """

    def __init__(
        self, pred_horizon: int, obs_horizon: int, action_horizon: int
    ) -> None:
        states, actions, episode_ends = load_push_t_zarr()

        self.normalization_ranges: Dict[str, NormalizationRange] = {
            "obs": NormalizationRange(values=states),
            "action": NormalizationRange(values=actions),
        }
        self._normalized = {
            "obs": self.normalization_ranges["obs"].normalize(states),
            "action": self.normalization_ranges["action"].normalize(actions),
        }

        self._indices = _create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1,
        )

        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        buffer_start, buffer_end, sample_start, sample_end = (
            int(v) for v in self._indices[idx]
        )
        nsample = _sample_sequence(
            train_data=self._normalized,
            sequence_length=self.pred_horizon,
            buffer_start=buffer_start,
            buffer_end=buffer_end,
            sample_start=sample_start,
            sample_end=sample_end,
        )
        nsample["obs"] = nsample["obs"][: self.obs_horizon, :]
        return nsample


def _create_sample_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    pad_before: int = 0,
    pad_after: int = 0,
) -> np.ndarray:
    """Enumerate ``(buffer_start, buffer_end, sample_start, sample_end)`` per window."""
    indices: List[List[int]] = []
    for i, end_idx in enumerate(episode_ends):
        start_idx = int(episode_ends[i - 1]) if i > 0 else 0
        end_idx = int(end_idx)
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        for idx in range(min_start, max_start + 1):
            buffer_start = max(idx, 0) + start_idx
            buffer_end = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end
            indices.append(
                [buffer_start, buffer_end, start_offset, sequence_length - end_offset]
            )
    return np.array(indices, dtype=np.int64)


def _sample_sequence(
    train_data: Dict[str, np.ndarray],
    sequence_length: int,
    buffer_start: int,
    buffer_end: int,
    sample_start: int,
    sample_end: int,
) -> Dict[str, np.ndarray]:
    """Slice a window out of the buffer, edge-padding as needed."""
    result: Dict[str, np.ndarray] = {}
    for key, arr in train_data.items():
        sample = arr[buffer_start:buffer_end]
        if sample_start == 0 and sample_end == sequence_length:
            result[key] = sample
            continue
        padded = np.zeros((sequence_length,) + arr.shape[1:], dtype=arr.dtype)
        if sample_start > 0:
            padded[:sample_start] = sample[0]
        if sample_end < sequence_length:
            padded[sample_end:] = sample[-1]
        padded[sample_start:sample_end] = sample
        result[key] = padded
    return result


class NormalizationRange:
    """Symmetric min/max normalizer mapping a feature vector to ``[-1, 1]``.

    Guards against zero-range features so constant dimensions stay finite.
    """

    def __init__(self, values: np.ndarray) -> None:
        flat = values.reshape(-1, values.shape[-1])
        self.minimum = flat.min(axis=0)
        self.maximum = flat.max(axis=0)

    def normalize(self, values: np.ndarray) -> np.ndarray:
        value_range = np.maximum(self.maximum - self.minimum, 1e-8)
        unit = (values - self.minimum) / value_range
        return unit * 2.0 - 1.0

    def denormalize(self, normalized_values: np.ndarray) -> np.ndarray:
        unit = (normalized_values + 1.0) / 2.0
        return unit * (self.maximum - self.minimum) + self.minimum
