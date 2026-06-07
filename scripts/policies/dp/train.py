import tempfile
from dataclasses import asdict
from pathlib import Path

import torch
import tyro
import wandb
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from rich.pretty import pretty_repr
from tqdm.auto import tqdm

from scripts.ashared import create_logger, log_device_info, set_seed
from scripts.evaluation.bryan.disturbance import DisturberConfig
from scripts.evaluation.bryan.evaluation import MaxRewardStatistics, evaluate_policy
from scripts.policies.dp.config import Config, DimensionalityConfig, HorizonLengthConfig
from scripts.policies.dp.dataset import PushTWindowDataset
from scripts.policies.dp.loss import _compute_denoising_loss
from scripts.policies.dp.network import ConditionalUnet1d
from scripts.policies.dp.noise_scheduler import get_noise_scheduler
from scripts.policies.dp.policy import load_diffusion_policy, save_diffusion_policy
from scripts.policies.shared.stat import Stat
from scripts.shared.data import load_demonstration_data_from_artifact
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH

logger = create_logger()


def main():
    config = tyro.cli(f=Config, config=[tyro.conf.UsePythonSyntaxForLiteralCollections])
    logger.info(pretty_repr(config))
    _run(config=config)


def _run(config: Config) -> None:
    set_seed(seed=config.seed)
    device = torch.device(device="cuda" if torch.cuda.is_available() else "cpu")
    log_device_info(device=device, logger=logger)

    condition_dimensionality = (
        config.dimensionality.observation * config.horizon_length.observation
    )

    noise_scheduler = get_noise_scheduler(
        diffusion_timestep_count=config.diffusion_timestep_count
    )

    noise_prediction_network = ConditionalUnet1d(
        input_channel_count=config.dimensionality.action,
        condition_dimensionality=condition_dimensionality,
        hidden_channel_counts=config.hidden_channel_counts,
    ).to(device=device)

    noise_prediction_network_ema = EMAModel(
        parameters=noise_prediction_network.parameters(),
        power=config.optimization.ema_power,
    )

    optimizer = torch.optim.AdamW(
        params=noise_prediction_network.parameters(),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )

    with (
        wandb.init(
            project="abc",
            job_type="train-diffusion-policy",
            config=asdict(config),
            dir=OUTPUT_DIRECTORY_PATH,
        ) as run,
        tempfile.TemporaryDirectory() as checkpoint_directory,
    ):
        checkpoint_directory_path = Path(checkpoint_directory)

        dataset = PushTWindowDataset(
            demonstration_data=load_demonstration_data_from_artifact(
                artifact=run.use_artifact(artifact_or_name=config.dataset_artifact),
                logger=logger,
            ),
            prediction_horizon_length=config.horizon_length.prediction,
            observation_horizon_length=config.horizon_length.observation,
            action_horizon_length=config.horizon_length.action,
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

        best_stat: Stat | None = None

        for epoch_number in tqdm(
            iterable=range(config.optimization.epoch_count),
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
                    observation_horizon_length=config.horizon_length.observation,
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
                        "training/loss": loss.item(),
                        "training/learning_rate": learning_rate_scheduler.get_last_lr()[
                            0
                        ],
                        "training/gradient_norm": gradient_norm.item(),
                    },
                    step=global_step,
                )
                global_step += 1

            periodic_eval_is_due = (
                epoch_number + 1
            ) % config.evaluation.periodic_interval_epoch_count == 0 or (
                epoch_number + 1
            ) == config.optimization.epoch_count
            if not periodic_eval_is_due:
                continue

            checkpoint_path = checkpoint_directory_path / "checkpoint.pt"
            _save_ema_checkpoint(
                path=checkpoint_path,
                network=noise_prediction_network,
                ema=noise_prediction_network_ema,
                dataset=dataset,
                horizon_length=config.horizon_length,
                dimensionality=config.dimensionality,
                condition_dimensionality=condition_dimensionality,
                hidden_channel_counts=config.hidden_channel_counts,
                diffusion_timestep_count=config.diffusion_timestep_count,
            )

            validation_result = _evaluate(
                checkpoint_path=checkpoint_path,
                episode_seeds=config.evaluation.episode_seeds,
                device=device,
            )
            run.log(
                data={
                    "validation/max_reward_mean": validation_result.mean,
                    "validation/max_reward_std": validation_result.std,
                },
                step=global_step,
            )

            if best_stat is None or validation_result.mean > best_stat.max_reward_mean:
                best_stat = Stat(
                    max_reward_mean=validation_result.mean,
                    epoch=epoch_number,
                    state_dict_path=checkpoint_path,
                )

        if best_stat is None:
            raise ValueError("No checkpoint was recorded")

        logger.info(
            "Selected best epoch %d (validation max reward %.4f).",
            best_stat.epoch,
            best_stat.max_reward_mean,
        )

        run.summary["selected/epoch"] = best_stat.epoch

        artifact = wandb.Artifact(name=f"{run.name}-policy", type="model")
        artifact.add_file(str(checkpoint_path))
        run.log_artifact(artifact)


@torch.no_grad()
def _evaluate(
    checkpoint_path: Path, episode_seeds: tuple[int, ...], device: torch.device
) -> MaxRewardStatistics:
    """Load a saved (EMA) checkpoint as a policy and run the shared rollout eval."""
    policy_factory = load_diffusion_policy(
        checkpoint_path=checkpoint_path, device=device
    )
    return evaluate_policy(
        policy_factory=policy_factory,
        episode_seeds=episode_seeds,
        disturber_config=None,
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


if __name__ == "__main__":
    main()
