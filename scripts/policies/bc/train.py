from dataclasses import dataclass

import torch
from torch.nn import Linear, ReLU, Sequential
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from scripts.policies.shared import POLICY_ROOT
from scripts.shared import (
    create_logger,
    initialize_run,
    load_push_t_zarr,
)

logger = create_logger()


@dataclass
class Config:
    seed: int
    epoch_count: int


def main():
    config, device, output_directory_path = initialize_run(
        config_class=Config,
        runs_output_directory_path=POLICY_ROOT / "bc",
        logger=logger,
    )

    policy = create_policy(epoch_count=config.epoch_count, device=device)

    state_dict_file_path = output_directory_path / "state_dict.pt"
    torch.save(obj=policy.state_dict(), f=state_dict_file_path)
    logger.info("Policy saved to %s", state_dict_file_path)


def create_policy(epoch_count: int, device: torch.device):
    model = Sequential(
        Linear(in_features=5, out_features=256),
        ReLU(),
        Linear(in_features=256, out_features=256),
        ReLU(),
        Linear(in_features=256, out_features=2),
    ).to(device=device)
    optimizer = Adam(params=model.parameters(), lr=1e-3)

    data_loader = DataLoader(
        dataset=PushTDataset(), batch_size=64, shuffle=True, pin_memory=device == "cuda"
    )

    for epoch in range(epoch_count):
        loss = training_iteration(
            model=model, optimizer=optimizer, data_loader=data_loader, device=device
        )
        logger.info("epoch %04d | loss %.6f", epoch, loss)

    return model


def training_iteration(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
):
    model.train()
    running_loss = 0.0
    batch_count = 0

    criterion = torch.nn.MSELoss()
    for batch_inputs, batch_labels in data_loader:
        device_batch_inputs = batch_inputs.to(device)
        device_batch_labels = batch_labels.to(device)
        optimizer.zero_grad()

        predictions = model(device_batch_inputs)

        batch_loss = criterion(predictions, device_batch_labels)

        batch_loss.backward()
        optimizer.step()

        running_loss += batch_loss.item()
        batch_count += 1

    return running_loss / batch_count


class PushTDataset(Dataset):
    def __init__(self):
        logger.info("Loading zarr dataset...")
        states, actions, _ = load_push_t_zarr()
        self.states = torch.from_numpy(states)
        self.actions = torch.from_numpy(actions)
        logger.info("Dataset loaded: %d timesteps", len(self.states))

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index: int):
        return self.states[index], self.actions[index]


if __name__ == "__main__":
    main()
