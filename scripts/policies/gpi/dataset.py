import numpy as np
import torch
from torch.utils.data import Dataset

from scripts.policies.gpi.geometry import se2_to_relative_action
from scripts.shared.data import DemonstrationData


class PushTEpisodeDataset(Dataset):
    """One demonstration episode per index"""

    def __init__(self, *, demonstration_data: DemonstrationData) -> None:
        states = demonstration_data.states
        raw_actions = demonstration_data.actions
        episode_ends = demonstration_data.episode_end_indices
        relative_actions = (
            se2_to_relative_action(
                states=torch.from_numpy(states), actions=torch.from_numpy(raw_actions)
            )
            .numpy()
            .astype(np.float32, copy=False)
        )

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

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        states, actions = self._episodes[index]
        return {
            "obs": self.normalize_observations(observations=states),
            "action": self.normalize_actions(actions=actions),
        }

    def normalize_observations(self, *, observations: np.ndarray) -> np.ndarray:
        return self.normalization_ranges["state"].normalize(values=observations)

    def normalize_actions(self, *, actions: np.ndarray) -> np.ndarray:
        return self.normalization_ranges["action"].normalize(values=actions)

    def denormalize_actions(self, *, normalized_actions: np.ndarray) -> np.ndarray:
        return self.normalization_ranges["action"].denormalize(values=normalized_actions)


class NormalizationRange:
    """Symmetric min/max normalizer mapping a feature vector to [-1, 1].

    Guards against zero-range features so constant dimensions stay finite.
    """

    def __init__(self, *, minimum: np.ndarray, maximum: np.ndarray) -> None:
        self.minimum = minimum
        self.maximum = maximum

    @classmethod
    def from_values(cls, *, values: np.ndarray) -> "NormalizationRange":
        flattened = values.reshape(-1, values.shape[-1])

        return cls(minimum=flattened.min(axis=0), maximum=flattened.max(axis=0))

    def normalize(self, *, values: np.ndarray) -> np.ndarray:
        value_range = np.clip(self.maximum - self.minimum, 1e-8, None)
        unit = (values - self.minimum) / value_range
        return unit * 2.0 - 1.0

    def denormalize(self, *, values: np.ndarray) -> np.ndarray:
        unit = (values + 1.0) / 2.0
        return unit * (self.maximum - self.minimum) + self.minimum
