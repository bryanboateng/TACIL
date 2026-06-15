import numpy as np
import torch
from torch.utils.data import Dataset

from scripts.policies.gpi.geometry import se2_to_relative_action
from scripts.shared.data import EpisodeDataset


class PushTEpisodeDataset(Dataset):
    def __init__(self, *, episode_dataset: EpisodeDataset) -> None:
        states = (
            episode_dataset.states
        )  # (total_timestep_count, observation_dimensionality)
        raw_actions = (
            episode_dataset.actions
        )  # (total_timestep_count, action_dimensionality)
        episode_ends = episode_dataset.episode_end_indices  # (episode_count,)
        relative_actions = (
            se2_to_relative_action(
                states=torch.from_numpy(states), actions=torch.from_numpy(raw_actions)
            )
            .numpy()
            .astype(np.float32, copy=False)
        )  # (total_timestep_count, action_dimensionality)

        self._episodes: list[tuple[np.ndarray, np.ndarray]] = []
        start = 0
        for end in episode_ends:
            end = int(end)
            self._episodes.append((states[start:end], relative_actions[start:end]))
            start = end

        self.normalization_ranges: dict[str, NormalizationRange] = {
            "state": NormalizationRange.from_values(values=states),
            "action": NormalizationRange.from_values(values=relative_actions),
        }

    def __len__(self) -> int:
        return len(self._episodes)

    def __getitem__(
        self, index: int
    ) -> dict[
        str, np.ndarray
    ]:  # "observation": (episode_length, observation_dimensionality), "action": (episode_length, action_dimensionality)
        states, actions = self._episodes[index]
        return {
            "observation": self.normalize_observations(observations=states),
            "action": self.normalize_actions(actions=actions),
        }

    def normalize_observations(
        self,
        *,
        observations: np.ndarray,  # (*, observation_dimensionality)
    ) -> np.ndarray:  # (*, observation_dimensionality) in [-1, 1]
        return self.normalization_ranges["state"].normalize(values=observations)

    def normalize_actions(
        self,
        *,
        actions: np.ndarray,  # (*, action_dimensionality)
    ) -> np.ndarray:  # (*, action_dimensionality) in [-1, 1]
        return self.normalization_ranges["action"].normalize(values=actions)

    def denormalize_actions(
        self,
        *,
        normalized_actions: np.ndarray,  # (*, action_dimensionality) in [-1, 1]
    ) -> np.ndarray:  # (*, action_dimensionality)
        return self.normalization_ranges["action"].denormalize(
            values=normalized_actions
        )


class NormalizationRange:
    """Symmetric min/max normalizer mapping a feature vector to [-1, 1].

    Guards against zero-range features so constant dimensions stay finite.
    """

    def __init__(
        self,
        *,
        minimum: np.ndarray,  # (dimensionality,)
        maximum: np.ndarray,  # (dimensionality,)
    ) -> None:
        self.minimum = minimum  # (dimensionality,)
        self.maximum = maximum  # (dimensionality,)

    @classmethod
    def from_values(
        cls,
        *,
        values: np.ndarray,  # (*, dimensionality) — any leading batch dimensions
    ) -> "NormalizationRange":
        flattened = values.reshape(
            -1, values.shape[-1]
        )  # (total_count, dimensionality)

        return cls(
            minimum=flattened.min(axis=0),  # (dimensionality,)
            maximum=flattened.max(axis=0),  # (dimensionality,)
        )

    def normalize(
        self,
        *,
        values: np.ndarray,  # (*, dimensionality)
    ) -> np.ndarray:  # (*, dimensionality) in [-1, 1]
        value_range = np.clip(self.maximum - self.minimum, 1e-8, None)
        unit = (values - self.minimum) / value_range
        return unit * 2.0 - 1.0

    def denormalize(
        self,
        *,
        values: np.ndarray,  # (*, dimensionality) in [-1, 1]
    ) -> np.ndarray:  # (*, dimensionality)
        unit = (values + 1.0) / 2.0
        return unit * (self.maximum - self.minimum) + self.minimum
