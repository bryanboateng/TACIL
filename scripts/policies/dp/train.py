import tempfile
from dataclasses import dataclass
from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium
import numpy as np
import torch
import torch.nn as nn
import tyro
import wandb
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from tqdm.auto import tqdm

from scripts.policies.dp.dataset import PushTWindowDataset
from scripts.policies.dp.network import ConditionalUnet1d
from scripts.policies.dp.noise_scheduler import get_noise_scheduler
from scripts.policies.dp.policy import (
    PolicyFactory,
    load_diffusion_policy,
    save_diffusion_policy,
)
from scripts.policies.shared import POLICY_ROOT
from scripts.shared import create_logger, log_device_info, set_seed

logger = create_logger()


@dataclass(frozen=True)
class DimensionalityConfig:
    observation: int = 5
    action: int = 2


@dataclass(frozen=True)
class HorizonLengthConfig:
    observation: int = 2
    prediction: int = 16
    action: int = 8


@dataclass(frozen=True)
class OptimizationConfig:
    batch_size: int = 256
    epoch_count: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    learning_rate_warmup_step_count: int = 500
    ema_power: float = 0.75


@dataclass(frozen=True)
class EvaluationConfig:
    periodic_interval_epoch_count: int = 25
    periodic_episode_seeds: tuple[int, ...] = tuple(range(100_000, 100_005))
    final_episode_seeds: tuple[int, ...] = tuple(range(1_000_000, 1_000_050))


def train(
    horizon_length: HorizonLengthConfig,
    optimization: OptimizationConfig,
    dimensionality: DimensionalityConfig,
    evaluation: EvaluationConfig,
    seed: int = 42,
    diffusion_timestep_count: int = 100,
    hidden_channel_counts: tuple[int, ...] = (256, 512, 1024),
) -> None:
    set_seed(seed=seed)
    device = torch.device(device="cuda" if torch.cuda.is_available() else "cpu")
    log_device_info(device=device, logger=logger)

    condition_dimensionality = dimensionality.observation * horizon_length.observation

    noise_scheduler = get_noise_scheduler(
        diffusion_timestep_count=diffusion_timestep_count
    )

    dataset = PushTWindowDataset(
        prediction_horizon_length=horizon_length.prediction,
        observation_horizon_length=horizon_length.observation,
        action_horizon_length=horizon_length.action,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=optimization.batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        num_workers=1,
        persistent_workers=True,
    )

    noise_prediction_network = ConditionalUnet1d(
        input_channel_count=dimensionality.action,
        condition_dimensionality=condition_dimensionality,
        hidden_channel_counts=hidden_channel_counts,
    ).to(device=device)

    noise_prediction_network_ema = EMAModel(
        parameters=noise_prediction_network.parameters(), power=optimization.ema_power
    )

    optimizer = torch.optim.AdamW(
        params=noise_prediction_network.parameters(),
        lr=optimization.learning_rate,
        weight_decay=optimization.weight_decay,
    )

    learning_rate_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=optimization.learning_rate_warmup_step_count,
        num_training_steps=len(dataloader) * optimization.epoch_count,
    )

    wandb.login()

    with wandb.init(
        project="asdf",
        job_type="diffusion_policy",
        config={
            "seed": seed,
            "diffusion_timestep_count": diffusion_timestep_count,
            "hidden_channel_counts": hidden_channel_counts,
            "observation_dimensionality": dimensionality.observation,
            "action_dimensionality": dimensionality.action,
            "observation_horizon_length": horizon_length.observation,
            "prediction_horizon_length": horizon_length.prediction,
            "action_horizon_length": horizon_length.action,
            "batch_size": optimization.batch_size,
            "epoch_count": optimization.epoch_count,
            "learning_rate": optimization.learning_rate,
            "weight_decay": optimization.weight_decay,
            "learning_rate_warmup_step_count": optimization.learning_rate_warmup_step_count,
            "ema_power": optimization.ema_power,
            "periodic_eval_interval_epoch_count": evaluation.periodic_interval_epoch_count,
            "periodic_eval_episode_seeds": list(evaluation.periodic_episode_seeds),
            "final_eval_episode_seeds": list(evaluation.final_episode_seeds),
        },
    ) as run:
        assert run.name is not None

        run_output_directory_path = POLICY_ROOT / "dp" / run.name
        run_output_directory_path.mkdir(parents=True, exist_ok=True)
        logger.info("Output directory: %s", run_output_directory_path)

        global_step = 0

        for epoch_number in tqdm(
            iterable=range(1, optimization.epoch_count + 1),
            desc="Epoch",
            unit="epoch",
            dynamic_ncols=True,
            position=0,
        ):
            run.log(data={"epoch": epoch_number}, step=global_step)
            noise_prediction_network.train()

            for training_batch in tqdm(
                iterable=dataloader,
                desc="Batch",
                unit="batch",
                leave=False,
                dynamic_ncols=True,
                position=1,
            ):
                loss = _compute_denoising_loss(
                    noise_prediction_network=noise_prediction_network,
                    training_batch=training_batch,
                    noise_scheduler=noise_scheduler,
                    observation_horizon_length=horizon_length.observation,
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
                        "train/loss": loss.item(),
                        "train/learning_rate": learning_rate_scheduler.get_last_lr()[0],
                        "train/gradient_norm": gradient_norm.item(),
                    },
                    step=global_step,
                )

                global_step += 1

            periodic_eval_is_due = (
                epoch_number % evaluation.periodic_interval_epoch_count == 0
                and epoch_number != optimization.epoch_count
            )
            if periodic_eval_is_due:
                intermediate_checkpoint_path = (
                    run_output_directory_path / f"checkpoint_epoch_{epoch_number}.pt"
                )
                _save_ema_checkpoint(
                    path=intermediate_checkpoint_path,
                    network=noise_prediction_network,
                    ema=noise_prediction_network_ema,
                    dataset=dataset,
                    horizon_length=horizon_length,
                    dimensionality=dimensionality,
                    condition_dimensionality=condition_dimensionality,
                    hidden_channel_counts=hidden_channel_counts,
                    diffusion_timestep_count=diffusion_timestep_count,
                )
                intermediate_artifact = wandb.Artifact(
                    name=f"{run.name}-checkpoint",
                    type="model",
                    metadata={"epoch": epoch_number, "global_step": global_step},
                )
                intermediate_artifact.add_file(str(intermediate_checkpoint_path))
                run.log_artifact(
                    intermediate_artifact, aliases=[f"epoch-{epoch_number}"]
                )
                logger.info(
                    "Intermediate checkpoint saved and logged to W&B at epoch %d",
                    epoch_number,
                )

                logger.info("Running periodic evaluation at epoch %d", epoch_number)
                max_coverage_mean, max_coverage_std = _evaluate(
                    network=noise_prediction_network,
                    ema=noise_prediction_network_ema,
                    dataset=dataset,
                    horizon_length=horizon_length,
                    dimensionality=dimensionality,
                    diffusion_timestep_count=diffusion_timestep_count,
                    hidden_channel_counts=hidden_channel_counts,
                    episode_seeds=evaluation.periodic_episode_seeds,
                    device=device,
                )
                run.log(
                    data={
                        "periodic_eval/max_coverage_mean": max_coverage_mean,
                        "periodic_eval/max_coverage_std": max_coverage_std,
                    },
                    step=global_step,
                )
                logger.info(
                    "Periodic eval at epoch %d: max coverage %.4f +/- %.4f",
                    epoch_number,
                    max_coverage_mean,
                    max_coverage_std,
                )

        checkpoint_file_path = run_output_directory_path / "checkpoint.pt"
        _save_ema_checkpoint(
            path=checkpoint_file_path,
            network=noise_prediction_network,
            ema=noise_prediction_network_ema,
            dataset=dataset,
            horizon_length=horizon_length,
            dimensionality=dimensionality,
            condition_dimensionality=condition_dimensionality,
            hidden_channel_counts=hidden_channel_counts,
            diffusion_timestep_count=diffusion_timestep_count,
        )
        logger.info("Policy saved to %s", checkpoint_file_path)

        artifact = wandb.Artifact(
            name=f"{run.name}-checkpoint",
            type="model",
            metadata={"epoch": optimization.epoch_count, "global_step": global_step},
        )
        artifact.add_file(str(checkpoint_file_path))
        run.log_artifact(
            artifact, aliases=["final", f"epoch-{optimization.epoch_count}"]
        )
        logger.info("Final checkpoint logged to W&B")

        logger.info(
            "Running final evaluation over %d episodes",
            len(evaluation.final_episode_seeds),
        )

        max_coverage_mean, max_coverage_std = _evaluate(
            network=noise_prediction_network,
            ema=noise_prediction_network_ema,
            dataset=dataset,
            horizon_length=horizon_length,
            dimensionality=dimensionality,
            diffusion_timestep_count=diffusion_timestep_count,
            hidden_channel_counts=hidden_channel_counts,
            episode_seeds=evaluation.final_episode_seeds,
            device=device,
        )
        run.summary["final_eval/max_coverage_mean"] = max_coverage_mean
        run.summary["final_eval/max_coverage_std"] = max_coverage_std
        logger.info(
            "Final eval complete: max coverage %.4f +/- %.4f",
            max_coverage_mean,
            max_coverage_std,
        )


def _save_ema_checkpoint(
    path: Path,
    network: ConditionalUnet1d,
    ema: EMAModel,
    dataset: PushTWindowDataset,
    horizon_length: HorizonLengthConfig,
    dimensionality: DimensionalityConfig,
    condition_dimensionality: int,
    hidden_channel_counts: tuple[int, ...],
    diffusion_timestep_count: int,
) -> None:
    """Save EMA weights to *path* without permanently modifying the network."""
    ema.store(parameters=network.parameters())
    ema.copy_to(parameters=network.parameters())
    try:
        save_diffusion_policy(
            path=path,
            noise_prediction_network=network,
            observation_normalization_range=dataset.observation_normalization_range,
            action_normalization_range=dataset.action_normalization_range,
            action_dimensionality=dimensionality.action,
            condition_dimensionality=condition_dimensionality,
            hidden_channel_counts=hidden_channel_counts,
            diffusion_timestep_count=diffusion_timestep_count,
            observation_horizon_length=horizon_length.observation,
            prediction_horizon_length=horizon_length.prediction,
            action_horizon_length=horizon_length.action,
        )
    finally:
        ema.restore(parameters=network.parameters())


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


@torch.no_grad()
def _evaluate(
    network: ConditionalUnet1d,
    ema: EMAModel,
    dataset: PushTWindowDataset,
    horizon_length: HorizonLengthConfig,
    dimensionality: DimensionalityConfig,
    diffusion_timestep_count: int,
    hidden_channel_counts: tuple[int, ...],
    episode_seeds: tuple[int, ...],
    device: torch.device,
) -> tuple[float, float]:
    ema.store(parameters=network.parameters())
    ema.copy_to(parameters=network.parameters())
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "eval_checkpoint.pt"
            condition_dimensionality = (
                dimensionality.observation * horizon_length.observation
            )
            save_diffusion_policy(
                path=checkpoint_path,
                noise_prediction_network=network,
                observation_normalization_range=dataset.observation_normalization_range,
                action_normalization_range=dataset.action_normalization_range,
                action_dimensionality=dimensionality.action,
                condition_dimensionality=condition_dimensionality,
                hidden_channel_counts=hidden_channel_counts,
                diffusion_timestep_count=diffusion_timestep_count,
                observation_horizon_length=horizon_length.observation,
                prediction_horizon_length=horizon_length.prediction,
                action_horizon_length=horizon_length.action,
            )

            policy_factory = load_diffusion_policy(
                checkpoint_path=checkpoint_path, device=device
            )

            maximum_coverages = np.array(
                run_evaluation_episodes(
                    episode_seeds=episode_seeds, policy_factory=policy_factory
                )
            )
            return maximum_coverages.mean().item(), maximum_coverages.std().item()
    finally:
        ema.restore(network.parameters())


def run_evaluation_episodes(
    episode_seeds: tuple[int, ...], policy_factory: PolicyFactory
) -> list[float]:
    environment = gymnasium.make(id="gym_pusht/PushT-v0")
    try:
        maximum_coverages: list[float] = []

        for episode_seed in tqdm(
            iterable=episode_seeds,
            desc="Episode",
            unit="episode",
            dynamic_ncols=True,
            position=2,
            leave=False,
        ):
            policy = policy_factory(episode_seed)

            observation, _ = environment.reset(seed=episode_seed)
            environment.action_space.seed(seed=episode_seed)
            environment.observation_space.seed(seed=episode_seed)

            maximum_coverage = 0.0

            while True:
                action = policy(observation)
                observation, _, terminated, truncated, info = environment.step(
                    action=action
                )
                maximum_coverage = max(maximum_coverage, float(info["coverage"]))
                if terminated or truncated:
                    break

            maximum_coverages.append(maximum_coverage)
    finally:
        environment.close()
    return maximum_coverages


if __name__ == "__main__":
    tyro.cli(f=train)
