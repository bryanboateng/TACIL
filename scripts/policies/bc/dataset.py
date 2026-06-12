import torch

from scripts.shared.data import DemonstrationData
from scripts.shared.normalization_range import NormalizationRange


class SingleStepDataset(torch.utils.data.Dataset):
    def __init__(self, *, demonstration_data: DemonstrationData) -> None:
        states = torch.from_numpy(demonstration_data.states).float()
        actions = torch.from_numpy(demonstration_data.actions).float()

        self.observation_normalization_range = NormalizationRange.from_values(
            values=states
        )
        self.action_normalization_range = NormalizationRange.from_values(values=actions)

        self._states = states
        self._actions = actions

    def __len__(self) -> int:
        return len(self._states)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._states[index], self._actions[index]
