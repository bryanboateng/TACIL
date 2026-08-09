import copy

import torch
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from tqdm.auto import tqdm
from wandb import Run

from scripts.evaluation.core.condition import Condition
from scripts.evaluation.core.scoring import (
    RewardStatistics,
    reward_statistics,
    run_scored_episodes,
)
from scripts.policies.dp.config import Config
from scripts.policies.dp.dataset import Dataset
from scripts.policies.dp.loss import _compute_denoising_loss
from scripts.policies.dp.network import ConditionalUnet1d
from scripts.policies.dp.noise_scheduler import get_ddpm_scheduler
from scripts.policies.dp.policy import build_diffusion_policy_factory
from scripts.policies.shared.checkpoint import Checkpoint
from scripts.shared.f import EpisodeStore
from scripts.shared.log import create_logger
from scripts.shared.policy_types import PolicyFactory
from scripts.shared.pusht_state import ACTION_DIMENSIONALITY, STATE_DIMENSIONALITY
from scripts.shared.seeding import set_seed

logger = create_logger()

# Every key this stage logs starts with it, so a stage run and a pipeline run
# report the same numbers under the same names. wandb groups panels by the part
# before the first slash, so the stage joins the group with a hyphen rather than
# a slash of its own.
STAGE_NAME = "3_policy"

# The x-axis of this stage's curves. The run may be shared with the rest of the
# pipeline, whose implicit step counter is not the optimization step.
STEP_METRIC_NAME = f"{STAGE_NAME}/step"


def train_diffusion_policy(
    *,
    config: Config,
    expert_episodes: EpisodeStore,
    augmented_episodes: EpisodeStore | None,
    condition: Condition,
    seed: int,
    device: torch.device,
    run: Run,
) -> ConditionalUnet1d:
    """Train on the recorded episodes plus whatever augmentation produced, and
    return the network holding the averaged weights that scored best."""
    set_seed(seed=seed)
    _define_step_metrics(run=run)

    condition_dimensionality = STATE_DIMENSIONALITY * config.horizon_length.observation

    noise_scheduler = get_ddpm_scheduler()

    augmented_transition_count = (
        0 if augmented_episodes is None else len(augmented_episodes.states)
    )
    run.summary[f"{STAGE_NAME}/transition_count"] = (
        len(expert_episodes.states) + augmented_transition_count
    )

    # The dataset measures the normalization ranges, so it comes first.
    dataset = Dataset(
        expert_episodes=expert_episodes,
        augmented_episodes=augmented_episodes,
        prediction_horizon_length=config.horizon_length.prediction,
        observation_horizon_length=config.horizon_length.observation,
        action_horizon_length=config.horizon_length.action,
    )

    noise_prediction_network = ConditionalUnet1d(
        input_channel_count=ACTION_DIMENSIONALITY,
        observation_dimensionality=STATE_DIMENSIONALITY,
        condition_dimensionality=condition_dimensionality,
        hidden_channel_counts=config.hidden_channel_counts,
        observation_normalizer=dataset.observation_normalizer,
        action_normalizer=dataset.action_normalizer,
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
            data={f"{STAGE_NAME}/epoch": epoch_number, STEP_METRIC_NAME: global_step}
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
                    f"{STAGE_NAME}/1_training/loss": loss.item(),
                    f"{STAGE_NAME}/1_training/learning_rate": (
                        learning_rate_scheduler.get_last_lr()[0]
                    ),
                    f"{STAGE_NAME}/1_training/gradient_norm": gradient_norm.item(),
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

        # What gets scored are the averaged weights, so they move into the
        # network for the duration of the evaluation and are taken out again.
        noise_prediction_network_ema.store(
            parameters=noise_prediction_network.parameters()
        )
        noise_prediction_network_ema.copy_to(
            parameters=noise_prediction_network.parameters()
        )
        try:
            ema_state_dict = copy.deepcopy(noise_prediction_network.state_dict())
            validation_reward = _evaluate(
                config=config,
                noise_prediction_network=noise_prediction_network,
                condition=condition,
                device=device,
            )
        finally:
            noise_prediction_network_ema.restore(
                parameters=noise_prediction_network.parameters()
            )
        run.log(
            data={
                f"{STAGE_NAME}/2_validation/reward_mean": validation_reward.mean,
                f"{STAGE_NAME}/2_validation/reward_std": validation_reward.std,
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

    run.summary[f"{STAGE_NAME}/3_selected/epoch"] = best_checkpoint.epoch
    run.summary[f"{STAGE_NAME}/3_selected/reward_mean"] = (
        best_checkpoint.validation_reward_mean
    )
    run.summary[f"{STAGE_NAME}/3_selected/reward_std"] = (
        best_checkpoint.validation_reward_std
    )

    noise_prediction_network.load_state_dict(best_checkpoint.state_dict)
    return noise_prediction_network


def build_policy_factory(
    *, config: Config, noise_prediction_network: ConditionalUnet1d, device: torch.device
) -> PolicyFactory:
    """The evaluation's view of a trained network. Kept here because the horizon
    lengths and the sampler come from this stage's config."""
    return build_diffusion_policy_factory(
        noise_prediction_network=noise_prediction_network,
        action_dimensionality=ACTION_DIMENSIONALITY,
        observation_horizon_length=config.horizon_length.observation,
        prediction_horizon_length=config.horizon_length.prediction,
        action_horizon_length=config.horizon_length.action,
        sampler=config.evaluation.sampler,
        device=device,
    )


def _define_step_metrics(*, run: Run) -> None:
    run.define_metric(name=STEP_METRIC_NAME)
    for metric_pattern in ("epoch", "1_training/*", "2_validation/*"):
        run.define_metric(
            name=f"{STAGE_NAME}/{metric_pattern}", step_metric=STEP_METRIC_NAME
        )


@torch.no_grad()
def _evaluate(
    *,
    config: Config,
    noise_prediction_network: ConditionalUnet1d,
    condition: Condition,
    device: torch.device,
) -> RewardStatistics:
    policy_factory = build_policy_factory(
        config=config, noise_prediction_network=noise_prediction_network, device=device
    )
    return reward_statistics(
        episode_records=run_scored_episodes(
            policy_factory=policy_factory,
            start_episode_seed=100_000,
            episode_count=config.evaluation.episode_count,
            condition=condition,
        )
    )
