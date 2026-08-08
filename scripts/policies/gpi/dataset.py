import numpy as np
import torch
from torch.utils.data import Dataset

from scripts.policies.gpi.geometry import se2_to_relative_action
from scripts.shared.f import EpisodeStore
from scripts.shared.normalization import MinMaxNormalizer


class PushTEpisodeDataset(Dataset):
    def __init__(self, *, episode_store: EpisodeStore) -> None:
        states = (
            episode_store.states
        )  # (total_timestep_count, observation_dimensionality)
        raw_actions = (
            episode_store.actions
        )  # (total_timestep_count, action_dimensionality)
        episode_ends = episode_store.episode_end_indices  # (episode_count,)
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

        self._observation_normalizer = MinMaxNormalizer.from_values(
            value_batch=torch.from_numpy(states)
        )
        self._action_normalizer = MinMaxNormalizer.from_values(
            value_batch=torch.from_numpy(relative_actions)
        )

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
        return _apply_numpy(
            normalizer=self._observation_normalizer.normalize, values=observations
        )

    def normalize_actions(
        self,
        *,
        actions: np.ndarray,  # (*, action_dimensionality)
    ) -> np.ndarray:  # (*, action_dimensionality) in [-1, 1]
        return _apply_numpy(
            normalizer=self._action_normalizer.normalize, values=actions
        )

    def denormalize_actions(
        self,
        *,
        normalized_actions: np.ndarray,  # (*, action_dimensionality) in [-1, 1]
    ) -> np.ndarray:  # (*, action_dimensionality)
        return _apply_numpy(
            normalizer=self._action_normalizer.denormalize, values=normalized_actions
        )


def _apply_numpy(
    *,
    normalizer,  # callable taking value_batch=Tensor and returning Tensor
    values: np.ndarray,  # (*, dimensionality)
) -> np.ndarray:  # (*, dimensionality)
    """Bridge the torch-based shared normalizer to GPI's numpy pipeline."""
    value_batch = torch.as_tensor(np.asarray(values), dtype=torch.float32)
    return normalizer(value_batch=value_batch).numpy()
