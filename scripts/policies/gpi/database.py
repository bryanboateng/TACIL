from typing import Iterable

import numpy as np
import torch

from scripts.policies.gpi.dataset import PushTEpisodeDataset
from scripts.policies.gpi.geometry import distance

Key = tuple[int, int]


class StateDatabase:
    """Dense GPU index over demonstrations providing fast GPI distance queries."""

    def __init__(
        self,
        *,
        dataset: PushTEpisodeDataset,
        device: str | None = None,
        subset_size: int | None = None,
    ) -> None:
        self.dataset = dataset
        self.device = torch.device(
            device=(
                device if device else ("cuda" if torch.cuda.is_available() else "cpu")
            )
        )
        self._build_full_buffers()
        self._active_indices = list(range(self._states_full.shape[0]))
        self._subset_size = subset_size
        self._sync_active_buffers()
        if subset_size is not None:
            self.resample_subset(subset_size=subset_size)

    def _build_full_buffers(self) -> None:
        states_per_ep: list[np.ndarray] = []
        actions_per_ep: list[np.ndarray] = []
        keys: list[Key] = []
        for episode_index in range(len(self.dataset)):
            episode = self.dataset[episode_index]
            normalized_observations = np.asarray(
                episode["observation"], dtype=np.float32
            )  # (episode_length, observation_dimensionality)
            normalized_actions = np.asarray(
                episode["action"], dtype=np.float32
            )  # (episode_length, action_dimensionality)
            states_per_ep.append(normalized_observations)
            actions_per_ep.append(normalized_actions)
            keys.extend(
                (episode_index, timestep)
                for timestep in range(len(normalized_observations))
            )

        all_states = np.concatenate(
            states_per_ep, axis=0
        )  # (total_timestep_count, observation_dimensionality)
        all_actions = np.concatenate(
            actions_per_ep, axis=0
        )  # (total_timestep_count, action_dimensionality)
        self._states_full = torch.from_numpy(all_states).to(
            device=self.device
        )  # (total_timestep_count, observation_dimensionality)
        self._actions_full = torch.from_numpy(all_actions).to(
            device=self.device
        )  # (total_timestep_count, action_dimensionality)
        self._keys_full: tuple[Key, ...] = tuple(keys)

    def resample_subset(self, *, subset_size: int | None = None) -> None:
        total = self._states_full.shape[0]
        size = subset_size if subset_size is not None else self._subset_size
        if size is None or size >= total:
            self._active_indices = list(range(total))
        else:
            choice = np.random.choice(total, size=size, replace=False)
            self._active_indices = sorted(int(i) for i in choice)
        self._sync_active_buffers()

    def _sync_active_buffers(self) -> None:
        if not self._active_indices:
            self._states = torch.empty(
                (0, self._states_full.shape[1]), device=self.device, dtype=torch.float32
            )  # (0, observation_dimensionality)
            self._actions = torch.empty(
                (0, self._actions_full.shape[1]),
                device=self.device,
                dtype=torch.float32,
            )  # (0, action_dimensionality)
            self._keys: list[Key] = []
        else:
            active = torch.tensor(
                self._active_indices, dtype=torch.long, device=self.device
            )  # (active_count,)
            self._states = self._states_full.index_select(
                dim=0, index=active
            )  # (active_count, observation_dimensionality)
            self._actions = self._actions_full.index_select(
                dim=0, index=active
            )  # (active_count, action_dimensionality)
            self._keys = [self._keys_full[i] for i in self._active_indices]
        self._key_to_active_index = {key: idx for idx, key in enumerate(self._keys)}

    @property
    def actions(self) -> torch.Tensor:  # (active_count, action_dimensionality)
        return self._actions

    def nearest(
        self,
        *,
        query: np.ndarray,  # (observation_dimensionality,)
        k: int = 1,
        exclude: Iterable[Key] | None = None,
    ) -> tuple[torch.Tensor, list[Key]]:  # distances (k,), keys list of length k
        """Return distances and keys of the top-k demonstrations (Alg.1, lines 4-10)."""
        num_active = len(self._keys)
        if num_active == 0:
            return torch.empty(0, device=self.device), []

        query_tensor = torch.as_tensor(
            query, dtype=torch.float32, device=self.device
        ).unsqueeze(dim=0)  # (1, observation_dimensionality)
        distances = distance(states=self._states, query=query_tensor)  # (active_count,)

        num_excluded = 0
        if exclude:
            excluded_indices = [
                self._key_to_active_index[key]
                for key in exclude
                if key in self._key_to_active_index
            ]
            if excluded_indices:
                distances = distances.clone()
                distances[excluded_indices] = float("inf")
                num_excluded = len(excluded_indices)

        available = num_active - num_excluded
        if available <= 0:
            return torch.empty(0, device=self.device), []

        values, indices = torch.topk(
            input=distances, k=min(k, available), largest=False
        )  # values (k,), indices (k,)
        selected_keys = [self._keys[i] for i in indices.cpu().tolist()]
        return values, selected_keys
