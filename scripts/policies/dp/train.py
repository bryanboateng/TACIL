import copy

import torch
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from torch import nn as nn
from tqdm.auto import tqdm
from wandb import Run

from scripts.evaluation.core.scoring import (
    calculate_reward_statistics,
    run_scored_episodes_over_conditions,
)
from scripts.environments.environment import Environment
from scripts.policies.dp.config import Config
from scripts.policies.dp.dataset import Dataset
from scripts.policies.dp.network import ConditionalUnet1d
from scripts.policies.dp.noise_scheduler import create_forward_noise_scheduler
from scripts.policies.dp.policy import create_diffusion_policy_factory
from scripts.policies.shared.checkpoint import Checkpoint
from scripts.policies.shared.observation_noise import disturb_half_of_the_batch
from scripts.shared import stage_names
from scripts.shared.episode_store import EpisodeStore
from scripts.shared.log import create_logger
from scripts.shared.seeding import set_seed

logger = create_logger()

STAGE_NAME = stage_names.POLICY

STEP_METRIC_NAME = f"{STAGE_NAME}/0_step"
EPOCH_METRIC_NAME = f"{STAGE_NAME}/0_epoch"

TRAINING_SECTION = f"{STAGE_NAME}/1_training"
VALIDATION_SECTION = f"{STAGE_NAME}/2_validation"


def train_diffusion_policy(
    *,
    config: Config,
    environment: Environment,
    expert_episodes: EpisodeStore,
    augmented_episodes: EpisodeStore | None,
    seed: int,
    device: torch.device,
    run: Run,
) -> ConditionalUnet1d:
    set_seed(seed=seed)
    _define_step_metrics(run=run)

    forward_noise_scheduler = create_forward_noise_scheduler()

    run.summary[f"{STAGE_NAME}/0_dataset/1_transition_count"] = len(
        expert_episodes.states
    ) + (0 if augmented_episodes is None else len(augmented_episodes.states))

    dataset = Dataset(
        expert_episodes=expert_episodes,
        augmented_episodes=augmented_episodes,
        horizon_lengths=config.horizon_lengths,
    )

    run.summary[f"{STAGE_NAME}/0_dataset/2_training_window_count"] = len(dataset)

    noise_prediction_network = ConditionalUnet1d(
        action_dimensionality=expert_episodes.actions.shape[1],
        condition_dimensionality=(
            expert_episodes.states.shape[1] * config.horizon_lengths.observation
        ),
        hidden_channel_counts=config.hidden_channel_counts,
        observation_normalizer=dataset.observation_normalizer,
        action_normalizer=dataset.action_normalizer,
    ).to(device=device)

    noise_prediction_network_ema = EMAModel(
        parameters=noise_prediction_network.parameters(),
        use_ema_warmup=True,
        inv_gamma=1.0,
        power=config.optimization.ema_power,
    )

    optimizer = torch.optim.AdamW(
        params=noise_prediction_network.parameters(),
        lr=config.optimization.learning_rate,
        # The original Diffusion Policy code uses 0.95 for the first moment, not Adam's usual 0.9.
        betas=(0.95, 0.999),
        weight_decay=config.optimization.weight_decay,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config.optimization.batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        num_workers=1,
        persistent_workers=True,
    )

    learning_rate_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=config.optimization.learning_rate_warmup_step_count,
        num_training_steps=len(dataloader) * config.optimization.epoch_count,
    )

    global_step = 0

    best_checkpoint: Checkpoint | None = None

    for epoch_index in tqdm(
        iterable=range(config.optimization.epoch_count),
        desc="Epoch",
        unit="epoch",
        dynamic_ncols=True,
        position=0,
    ):
        noise_prediction_network.train()

        for training_batch in tqdm(
            iterable=dataloader,
            desc="Batch",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
            position=1,
        ):
            loss = _calculate_denoising_loss(
                noise_prediction_network=noise_prediction_network,
                training_batch=training_batch,
                forward_noise_scheduler=forward_noise_scheduler,
                observation_horizon_length=config.horizon_lengths.observation,
                observation_noise=config.observation_noise,
                device=device,
            )

            loss.backward()

            gradient_norm = torch.norm(
                input=torch.stack(
                    tensors=[
                        parameter.grad.norm()
                        for parameter in noise_prediction_network.parameters()
                        if parameter.grad is not None
                    ]
                )
            )

            optimizer.step()
            optimizer.zero_grad()
            learning_rate_scheduler.step()
            noise_prediction_network_ema.step(noise_prediction_network.parameters())

            run.log(
                data={
                    f"{TRAINING_SECTION}/1_loss": loss.item(),
                    f"{TRAINING_SECTION}/3_learning_rate": (
                        learning_rate_scheduler.get_last_lr()[0]
                    ),
                    f"{TRAINING_SECTION}/2_gradient_norm": gradient_norm.item(),
                    STEP_METRIC_NAME: global_step,
                }
            )
            global_step += 1

        if (epoch_index + 1) % config.validation.interval_epoch_count != 0:
            continue

        # What gets scored are the averaged weights, so they move into the
        # network for the duration of the validation and are taken out again.
        noise_prediction_network_ema.store(
            parameters=noise_prediction_network.parameters()
        )
        noise_prediction_network_ema.copy_to(
            parameters=noise_prediction_network.parameters()
        )
        try:
            ema_state_dict = copy.deepcopy(noise_prediction_network.state_dict())
            with torch.no_grad():
                validation_reward = calculate_reward_statistics(
                    episode_records=run_scored_episodes_over_conditions(
                        environment=environment,
                        policy_factory=create_diffusion_policy_factory(
                            noise_prediction_network=noise_prediction_network,
                            horizon_lengths=config.horizon_lengths,
                            noise_scheduler_name=config.sampler,
                            device=device,
                        ),
                        episode_count_per_condition=config.validation.episode_count_per_condition,
                    )
                )
        finally:
            noise_prediction_network_ema.restore(
                parameters=noise_prediction_network.parameters()
            )
        run.log(
            data={
                f"{VALIDATION_SECTION}/1_reward_mean": validation_reward.mean,
                f"{VALIDATION_SECTION}/2_reward_std": validation_reward.std,
                EPOCH_METRIC_NAME: epoch_index,
                STEP_METRIC_NAME: global_step,
            }
        )

        if (
            best_checkpoint is None
            or validation_reward.mean > best_checkpoint.validation_reward_mean
        ):
            best_checkpoint = Checkpoint(
                validation_reward_mean=validation_reward.mean,
                validation_reward_std=validation_reward.std,
                epoch=epoch_index,
                state_dict=ema_state_dict,
            )

    if best_checkpoint is None:
        raise ValueError("No checkpoint was recorded")

    run.summary[f"{STAGE_NAME}/3_best_checkpoint/1_epoch"] = best_checkpoint.epoch
    run.summary[f"{STAGE_NAME}/3_best_checkpoint/2_validation_reward_mean"] = (
        best_checkpoint.validation_reward_mean
    )
    run.summary[f"{STAGE_NAME}/3_best_checkpoint/3_validation_reward_std"] = (
        best_checkpoint.validation_reward_std
    )

    noise_prediction_network.load_state_dict(best_checkpoint.state_dict)
    return noise_prediction_network


# ---- Private --------------------------------------------


def _define_step_metrics(*, run: Run) -> None:
    run.define_metric(name=STEP_METRIC_NAME)
    for name in (EPOCH_METRIC_NAME, f"{TRAINING_SECTION}/*", f"{VALIDATION_SECTION}/*"):
        run.define_metric(name=name, step_metric=STEP_METRIC_NAME)


def _calculate_denoising_loss(
    *,
    noise_prediction_network: ConditionalUnet1d,
    training_batch: tuple[torch.Tensor, torch.Tensor],
    forward_noise_scheduler: DDPMScheduler,
    observation_horizon_length: int,
    observation_noise: float | None,
    device: torch.device,
) -> torch.Tensor:
    observations, actions = training_batch
    observations = observations.to(
        device=device
    )  # (batch_size, observation_horizon_length, observation_dimensionality)
    actions = actions.to(
        device=device
    )  # (batch_size, prediction_horizon_length, action_dimensionality)

    batch_size = observations.shape[0]

    # FiLM conditioning: flatten observation history into a single vector per example.
    condition_batch = observations[:, :observation_horizon_length, :].flatten(
        start_dim=1
    )  # (batch_size, observation_horizon_length * observation_dimensionality)

    if observation_noise is not None:
        condition_batch = disturb_half_of_the_batch(
            value_batch=condition_batch, noise=observation_noise
        )

    noise_batch = torch.randn(
        size=actions.shape, device=device
    )  # (batch_size, prediction_horizon_length, action_dimensionality)
    diffusion_timestep_batch = torch.randint(
        low=0,
        high=forward_noise_scheduler.config["num_train_timesteps"],
        size=(batch_size,),
        device=device,
    )  # (batch_size,)
    noisy_action_batch = forward_noise_scheduler.add_noise(
        original_samples=actions,
        noise=noise_batch,
        # Ignore because PyTorch returns Tensors but diffusers expects IntTensor in its type hints.
        timesteps=diffusion_timestep_batch,  # type: ignore[arg-type] # ty:ignore[invalid-argument-type]
    )  # (batch_size, prediction_horizon_length, action_dimensionality)

    # The target is the noise as it was drawn, before the timestep scaled it.
    predicted_noise_batch = noise_prediction_network(
        noisy_sample_batch=noisy_action_batch,
        diffusion_timestep=diffusion_timestep_batch,
        condition_batch=condition_batch,
    )  # (batch_size, prediction_horizon_length, action_dimensionality)
    return nn.functional.mse_loss(input=predicted_noise_batch, target=noise_batch)
