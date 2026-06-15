import torch

from scripts.shared.data import EpisodeDataset


class SingleStepDataset(torch.utils.data.Dataset):
    def __init__(self, *, episode_dataset: EpisodeDataset) -> None:
        self.states = torch.from_numpy(
            episode_dataset.states
        ).float()  # (total_timestep_count, observation_dimensionality)

        self.actions = torch.from_numpy(
            episode_dataset.actions
        ).float()  # (total_timestep_count, action_dimensionality)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(
        self, index: int
    ) -> tuple[
        torch.Tensor, torch.Tensor
    ]:  # (observation_dimensionality,), (action_dimensionality,)
        return self.states[index], self.actions[index]
