import torch

from scripts.shared.state_action_pairs import StateActionPairs


class Dataset(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, *, state_action_pairs: StateActionPairs) -> None:
        self.states = torch.from_numpy(
            state_action_pairs.states
        ).float()  # (transition_count, state_dimensionality)

        self.actions = torch.from_numpy(
            state_action_pairs.actions
        ).float()  # (transition_count, action_dimensionality)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(
        self, index: int
    ) -> tuple[
        torch.Tensor, torch.Tensor
    ]:  # (state_dimensionality,), (action_dimensionality,)
        return self.states[index], self.actions[index]
