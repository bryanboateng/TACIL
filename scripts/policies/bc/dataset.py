import torch

from scripts.shared.data import EpisodeDataset
from scripts.shared.normalization_range import NormalizationRange


class SingleStepDataset(torch.utils.data.Dataset):
    def __init__(self, *, episode_dataset: EpisodeDataset) -> None:
        states = torch.from_numpy(
            episode_dataset.states
        ).float()  # (total_timestep_count, observation_dimensionality)

        actions = torch.from_numpy(
            episode_dataset.actions
        ).float()  # (total_timestep_count, action_dimensionality)

        self.observation_normalization_range = NormalizationRange.from_values(
            values=states
        )
        self.action_normalization_range = NormalizationRange.from_values(values=actions)

        self._states = states
        self._actions = actions

    def __len__(self) -> int:
        return len(self._states)

    def __getitem__(
        self, index: int
    ) -> tuple[
        torch.Tensor, torch.Tensor
    ]:  # (observation_dimensionality,), (action_dimensionality,)
        return self._states[index], self._actions[index]
