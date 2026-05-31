import copy

import torch


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric: float = float("inf")
        self.best_state_dict: dict[str, torch.Tensor] | None = None
        self.best_epoch: int = -1
        self._evaluations_without_improvement = 0

    def update(self, metric: float, model: torch.nn.Module, epoch: int) -> bool:
        if metric < self.best_metric - self.min_delta:
            self.best_metric = metric
            self.best_state_dict = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self._evaluations_without_improvement = 0
        else:
            self._evaluations_without_improvement += 1
        return self._evaluations_without_improvement >= self.patience
