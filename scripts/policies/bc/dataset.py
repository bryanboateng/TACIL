import torch

from scripts.shared.data import EpisodeDataset


class Dataset(torch.utils.data.Dataset):
    def __init__(self, *, episode_dataset: EpisodeDataset) -> None:
        self.states = torch.from_numpy(
            episode_dataset.states
        ).float()  # (total_timestep_count, state_dimensionality)

        self.actions = torch.from_numpy(
            episode_dataset.actions
        ).float()  # (total_timestep_count, action_dimensionality)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(
        self, index: int
    ) -> tuple[
        torch.Tensor, torch.Tensor
    ]:  # (state_dimensionality,), (action_dimensionality,)
        return self.states[index], self.actions[index]
