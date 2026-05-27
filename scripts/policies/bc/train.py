import torch
import tyro
import wandb
from torch.nn import Linear, ReLU, Sequential
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from scripts.policies.shared import POLICY_ROOT
from scripts.shared import create_logger, load_push_t_zarr, log_device_info, set_seed

logger = create_logger()


def main(seed: int, epoch_count: int):
    set_seed(seed=seed)
    device = torch.device(device="cuda" if torch.cuda.is_available() else "cpu")
    log_device_info(device=device, logger=logger)

    data_loader = DataLoader(
        dataset=PushTDataset(),
        batch_size=64,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )

    model = Sequential(
        Linear(in_features=5, out_features=256),
        ReLU(),
        Linear(in_features=256, out_features=256),
        ReLU(),
        Linear(in_features=256, out_features=2),
    ).to(device=device)

    optimizer = Adam(params=model.parameters(), lr=1e-3)

    wandb.login()

    with wandb.init(
        project="asdf",
        job_type="behaviour_cloning",
        config={"seed": seed, "epoch_count": epoch_count},
    ) as run:
        assert run.name is not None

        run_output_directory_path = POLICY_ROOT / "bc" / run.name
        run_output_directory_path.mkdir(parents=True, exist_ok=True)

        logger.info("Output directory: %s", run_output_directory_path)

        for _ in tqdm(range(epoch_count), desc="Epoch"):
            model.train()
            criterion = torch.nn.MSELoss()
            for batch_inputs, batch_labels in data_loader:
                device_batch_inputs = batch_inputs.to(device)
                device_batch_labels = batch_labels.to(device)
                optimizer.zero_grad()

                predictions = model(device_batch_inputs)

                batch_loss = criterion(predictions, device_batch_labels)

                batch_loss.backward()

                gradient_norm = torch.norm(
                    input=torch.stack(
                        tensors=[
                            parameter.grad.norm()
                            for parameter in model.parameters()
                            if parameter.grad is not None
                        ]
                    )
                )

                optimizer.step()

                run.log(
                    {
                        "train/loss": batch_loss.item(),
                        "train/gradient_norm": gradient_norm.item(),
                    }
                )

        state_dict_file_path = run_output_directory_path / "state_dict.pt"
        torch.save(obj=model.state_dict(), f=state_dict_file_path)
        logger.info("Policy saved to %s", state_dict_file_path)

        artifact = wandb.Artifact(name=f"{run.name}-checkpoint", type="model")
        artifact.add_file(str(state_dict_file_path))
        run.log_artifact(artifact)


class PushTDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
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
    tyro.cli(f=main)
