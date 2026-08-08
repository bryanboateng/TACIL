import torch

from scripts.shared.data import TransitionBatch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, *, transitions: TransitionBatch) -> None:
        self.states = torch.from_numpy(
            transitions.states
        ).float()  # (transition_count, state_dimensionality)

        self.actions = torch.from_numpy(
            transitions.actions
        ).float()  # (transition_count, action_dimensionality)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(
        self, index: int
    ) -> tuple[
        torch.Tensor, torch.Tensor
    ]:  # (state_dimensionality,), (action_dimensionality,)
        return self.states[index], self.actions[index]
