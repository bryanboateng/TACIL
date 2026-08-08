import copy
from pathlib import Path
from typing import Literal

import torch
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from tqdm.auto import tqdm
from wandb import Run

from scripts.data.base.episode_store import EpisodeStore
from scripts.evaluation.core.condition import Condition
from scripts.evaluation.core.scoring import (
    RewardStatistics,
    reward_statistics,
    run_scored_episodes,
)
from scripts.policies.dp.config import Config, DimensionalityConfig, HorizonLengthConfig
from scripts.policies.dp.dataset import Dataset
from scripts.policies.dp.loss import _compute_denoising_loss
from scripts.policies.dp.network import ConditionalUnet1d
from scripts.policies.dp.noise_scheduler import get_ddpm_scheduler
from scripts.policies.dp.policy import load_diffusion_policy, save_diffusion_policy
from scripts.policies.shared.checkpoint import Checkpoint
from scripts.shared.log import create_logger
from scripts.shared.seeding import set_seed

logger = create_logger()

# Every key this stage logs starts with it, so a stage run and a pipeline run
# report the same numbers under the same names. wandb groups panels by the part
# before the first slash, so the stage joins the group with a hyphen rather than
# a slash of its own.
STAGE_PREFIX = "3_policy-"

# The x-axis of this stage's curves. The run may be shared with the rest of the
# pipeline, whose implicit step counter is not the optimization step.
STEP_METRIC_NAME = f"{STAGE_PREFIX}step"


def train_diffusion_policy(
    *,
    config: Config,
    expert_episodes: EpisodeStore,
    augmented_episodes: EpisodeStore | None,
    checkpoint_directory_path: Path,
    device: torch.device,
    run: Run,
) -> Path:
    """Train on the recorded episodes plus whatever augmentation produced,
    keeping the periodically evaluated EMA checkpoint that scored best. It is
    written into *checkpoint_directory_path*, whose lifetime the caller owns.
    """
    set_seed(seed=config.seed)
    _define_step_metrics(run=run)

    condition_dimensionality = (
        config.dimensionality.observation * config.horizon_length.observation
    )

    noise_scheduler = get_ddpm_scheduler()

    noise_prediction_network = ConditionalUnet1d(
        input_channel_count=config.dimensionality.action,
        observation_dimensionality=config.dimensionality.observation,
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

    dataset = Dataset(
        expert_episodes=expert_episodes,
        augmented_episodes=augmented_episodes,
        prediction_horizon_length=config.horizon_length.prediction,
        observation_horizon_length=config.horizon_length.observation,
        action_horizon_length=config.horizon_length.action,
    )

    noise_prediction_network.initialize_normalizers(
        observation_normalizer=dataset.observation_normalizer,
        action_normalizer=dataset.action_normalizer,
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

    for epoch_number in tqdm(
        iterable=range(config.optimization.epoch_count),
        desc="Epoch",
        unit="epoch",
        dynamic_ncols=True,
        position=0,
    ):
        run.log(
            data={f"{STAGE_PREFIX}epoch": epoch_number, STEP_METRIC_NAME: global_step}
        )
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
                    f"{STAGE_PREFIX}training/loss": loss.item(),
                    f"{STAGE_PREFIX}training/learning_rate": (
                        learning_rate_scheduler.get_last_lr()[0]
                    ),
                    f"{STAGE_PREFIX}training/gradient_norm": gradient_norm.item(),
                    STEP_METRIC_NAME: global_step,
                }
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
        ema_state_dict = _save_ema_checkpoint(
            path=checkpoint_path,
            network=noise_prediction_network,
            ema=noise_prediction_network_ema,
            horizon_length=config.horizon_length,
            dimensionality=config.dimensionality,
            condition_dimensionality=condition_dimensionality,
            hidden_channel_counts=config.hidden_channel_counts,
        )

        validation_reward = _evaluate(
            checkpoint_path=checkpoint_path,
            episode_count=config.evaluation.episode_count,
            sampler=config.evaluation.sampler,
            condition=Condition(
                floor_condition=config.evaluation.floor_condition,
                disturbance_level=config.evaluation.disturbance_level,
            ),
            device=device,
        )
        run.log(
            data={
                f"{STAGE_PREFIX}validation/reward_mean": validation_reward.mean,
                f"{STAGE_PREFIX}validation/reward_std": validation_reward.std,
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
                epoch=epoch_number,
                state_dict=ema_state_dict,
            )

    if best_checkpoint is None:
        raise ValueError("No checkpoint was recorded")

    logger.info(
        "Selected best epoch %d (validation max reward %.4f).",
        best_checkpoint.epoch,
        best_checkpoint.validation_reward_mean,
    )

    run.summary[f"{STAGE_PREFIX}selected/epoch"] = best_checkpoint.epoch
    run.summary[f"{STAGE_PREFIX}selected/reward_mean"] = (
        best_checkpoint.validation_reward_mean
    )
    run.summary[f"{STAGE_PREFIX}selected/reward_std"] = (
        best_checkpoint.validation_reward_std
    )

    noise_prediction_network.load_state_dict(best_checkpoint.state_dict)
    final_path = checkpoint_directory_path / "checkpoint.pt"
    save_diffusion_policy(
        path=final_path,
        noise_prediction_network=noise_prediction_network,
        action_dimensionality=config.dimensionality.action,
        condition_dimensionality=condition_dimensionality,
        hidden_channel_counts=config.hidden_channel_counts,
        observation_horizon_length=config.horizon_length.observation,
        prediction_horizon_length=config.horizon_length.prediction,
        action_horizon_length=config.horizon_length.action,
    )
    return final_path


def _define_step_metrics(*, run: Run) -> None:
    run.define_metric(name=STEP_METRIC_NAME)
    for metric_pattern in ("epoch", "training/*", "validation/*"):
        run.define_metric(
            name=f"{STAGE_PREFIX}{metric_pattern}", step_metric=STEP_METRIC_NAME
        )


@torch.no_grad()
def _evaluate(
    *,
    checkpoint_path: Path,
    episode_count: int,
    sampler: Literal["ddim", "ddpm"],
    condition: Condition,
    device: torch.device,
) -> RewardStatistics:
    policy_factory = load_diffusion_policy(
        checkpoint_path=checkpoint_path, sampler=sampler, device=device
    )
    return reward_statistics(
        episode_records=run_scored_episodes(
            policy_factory=policy_factory,
            start_episode_seed=100_000,
            episode_count=episode_count,
            condition=condition,
        )
    )


def _save_ema_checkpoint(
    *,
    path: Path,
    network: ConditionalUnet1d,
    ema: EMAModel,
    horizon_length: HorizonLengthConfig,
    dimensionality: DimensionalityConfig,
    condition_dimensionality: int,
    hidden_channel_counts: tuple[int, ...],
) -> dict:
    """Save EMA weights to *path* without permanently modifying the network.

    Returns a deep copy of the EMA state dict so the caller can keep the best
    checkpoint in memory without re-reading the file.
    """
    ema.store(parameters=network.parameters())
    ema.copy_to(parameters=network.parameters())
    try:
        state_dict = copy.deepcopy(network.state_dict())
        save_diffusion_policy(
            path=path,
            noise_prediction_network=network,
            action_dimensionality=dimensionality.action,
            condition_dimensionality=condition_dimensionality,
            hidden_channel_counts=hidden_channel_counts,
            observation_horizon_length=horizon_length.observation,
            prediction_horizon_length=horizon_length.prediction,
            action_horizon_length=horizon_length.action,
        )
        return state_dict
    finally:
        ema.restore(parameters=network.parameters())
