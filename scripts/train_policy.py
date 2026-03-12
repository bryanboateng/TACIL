import logging
from pathlib import Path

import torch
import zarr
from torch.nn import Linear, ReLU, Sequential
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting training")
    policy = create_policy()

    save_path = Path("policy_model_state.pth")
    torch.save(obj=policy.state_dict(), f=save_path)
    logger.info("Policy saved to %s", save_path)


def create_policy():
    model = Sequential(
        Linear(in_features=5, out_features=256),
        ReLU(),
        Linear(in_features=256, out_features=256),
        ReLU(),
        Linear(in_features=256, out_features=2),
    )
    optimizer = Adam(params=model.parameters(), lr=1e-3)

    data_loader = DataLoader(dataset=PushTDataset(), batch_size=64, shuffle=True)

    for epoch in range(400):
        loss = training_iteration(
            model=model, optimizer=optimizer, data_loader=data_loader
        )
        logger.info("epoch %04d | loss %.6f", epoch, loss)

    return model


def training_iteration(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: torch.utils.data.DataLoader,
) -> float:
    model.train()
    running_loss = 0.0
    batch_count = 0

    criterion = torch.nn.MSELoss()
    for batch_inputs, batch_labels in data_loader:
        optimizer.zero_grad()

        predictions = model(batch_inputs)

        batch_loss = criterion(predictions, batch_labels)

        batch_loss.backward()
        optimizer.step()

        running_loss += batch_loss.item()
        batch_count += 1

    return running_loss / batch_count


class PushTDataset(Dataset):
    def __init__(self):
        logger.info("Loading Zarr dataset...")
        store = zarr.open(
            store=Path("../expert_data/pusht_cchi_v7_replay.zarr"), mode="r"
        )
        self.states = torch.tensor(store["data/state"][:], dtype=torch.float32)
        self.actions = torch.tensor(store["data/action"][:], dtype=torch.float32)
        logger.info("Dataset loaded: %d timesteps", len(self.states))

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        return self.states[index], self.actions[index]


if __name__ == "__main__":
    main()
