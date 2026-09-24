import torch

from scripts.shared.state_action_transitions import StateActionTransitions


class Dataset(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, *, state_action_transitions: StateActionTransitions) -> None:
        self.states = torch.from_numpy(
            state_action_transitions.states
        ).float()  # (pair_count, state_dimensionality)

        self.actions = torch.from_numpy(
            state_action_transitions.actions
        ).float()  # (pair_count, action_dimensionality)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(
        self, index: int
    ) -> tuple[
        torch.Tensor, torch.Tensor
    ]:  # (state_dimensionality,), (action_dimensionality,)
        return self.states[index], self.actions[index]
