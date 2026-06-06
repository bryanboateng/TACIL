import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import nn as nn

from scripts.policies.dp.network import ConditionalUnet1d


def _compute_denoising_loss(
    noise_prediction_network: ConditionalUnet1d,
    training_batch: tuple[torch.Tensor, torch.Tensor],
    noise_scheduler: DDPMScheduler,
    observation_horizon_length: int,
    device: torch.device,
) -> torch.Tensor:
    observations, actions = training_batch
    observations = observations.to(device=device)
    actions = actions.to(device=device)

    batch_size = observations.shape[0]

    # FiLM conditioning: flatten observation history into
    # (batch_size, observation_horizon_length * observation_dimensionality).
    condition_batch = observations[:, :observation_horizon_length, :].flatten(
        start_dim=1
    )

    noise_batch = torch.randn(size=actions.shape, device=device)
    diffusion_timestep_batch = torch.randint(
        low=0,
        high=noise_scheduler.config["num_train_timesteps"],
        size=(batch_size,),
        device=device,
    )
    noisy_action_batch = noise_scheduler.add_noise(
        original_samples=actions,
        noise=noise_batch,
        # Ignore because PyTorch returns Tensors but diffusers expects IntTensor in its type hints.
        timesteps=diffusion_timestep_batch,  # type: ignore[arg-type] # ty:ignore[invalid-argument-type]
    )

    noise_prediction_batch = noise_prediction_network(
        noisy_sample_batch=noisy_action_batch,
        diffusion_timestep=diffusion_timestep_batch,
        condition_batch=condition_batch,
    )
    return nn.functional.mse_loss(input=noise_prediction_batch, target=noise_batch)
