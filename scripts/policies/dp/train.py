"""Training entrypoint for the PushT diffusion policy."""

import numpy as np
import torch
import torch.nn as nn
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from tqdm.auto import tqdm

from scripts.policies.dp.config import DiffusionPolicyConfig
from scripts.policies.dp.dataset import PushTWindowDataset
from scripts.policies.dp.network import ConditionalUnet1D
from scripts.policies.shared import POLICY_ROOT
from scripts.shared import create_logger, initialize_run

logger = create_logger()


def build_noise_scheduler(config: DiffusionPolicyConfig) -> DDPMScheduler:
    return DDPMScheduler(
        num_train_timesteps=config.num_diffusion_steps,
        # squared-cosine schedule gave the best results in our experiments
        beta_schedule="squaredcos_cap_v2",
        # clip output to [-1, 1] for stability
        clip_sample=True,
        # network predicts noise, not the denoised action
        prediction_type="epsilon",
    )


def train() -> None:
    config, device, output_directory_path = initialize_run(
        config_class=DiffusionPolicyConfig,
        runs_output_directory_path=POLICY_ROOT / "dp",
        logger=logger,
    )

    noise_scheduler = build_noise_scheduler(config)

    dataset = PushTWindowDataset(
        pred_horizon=config.prediction_horizon,
        obs_horizon=config.observation_horizon,
        action_horizon=config.action_horizon,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_dataloader_workers,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )

    net = ConditionalUnet1D(
        input_dim=config.action_dim,
        global_cond_dim=config.global_cond_dim,
    ).to(device)

    ema = EMAModel(parameters=net.parameters(), power=config.ema_power)
    optimizer = torch.optim.AdamW(
        params=net.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=len(dataloader) * config.num_epochs,
    )

    for _ in tqdm(range(config.num_epochs), desc="Epoch"):
        epoch_losses = []
        for nbatch in tqdm(dataloader, desc="Batch", leave=False):
            loss = _training_step(
                net=net,
                batch=nbatch,
                noise_scheduler=noise_scheduler,
                obs_horizon=config.observation_horizon,
                device=device,
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            ema.step(net.parameters())
            epoch_losses.append(loss.item())
        tqdm.write(f"epoch loss: {np.mean(epoch_losses):.6f}")

    # Save the EMA weights (preferred for inference).
    ema.copy_to(net.parameters())
    state_dict_file_path = output_directory_path / "state_dict.pt"
    torch.save(obj=net.state_dict(), f=state_dict_file_path)
    logger.info("Policy saved to %s", state_dict_file_path)


def _training_step(
    net: ConditionalUnet1D,
    batch: dict,
    noise_scheduler: DDPMScheduler,
    obs_horizon: int,
    device: torch.device,
) -> torch.Tensor:
    nobs = batch["obs"].to(device)
    naction = batch["action"].to(device)
    batch_size = nobs.shape[0]

    # FiLM conditioning: flatten observation history into (B, obs_horizon * obs_dim).
    obs_cond = nobs[:, :obs_horizon, :].flatten(start_dim=1)

    noise = torch.randn(naction.shape, device=device)
    timesteps = torch.randint(
        low=0,
        high=noise_scheduler.config.num_train_timesteps,
        size=(batch_size,),
        device=device,
    ).long()

    noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)
    noise_pred = net(noisy_actions, timesteps, global_cond=obs_cond)
    return nn.functional.mse_loss(noise_pred, noise)


if __name__ == "__main__":
    train()
