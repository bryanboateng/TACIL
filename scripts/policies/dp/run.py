"""Diffusion policy, optionally on TaCIL-augmented data: one trial end to end.

The augmentation runs inside this process and its output never becomes an
artifact. Every trial searches parameters that change that output, so there
would be nothing to reuse it for.

CCIL is not offered here: its labels are single (state, action) pairs, which
cannot fill a window.
"""

import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated

import torch
import tyro
import wandb
from rich.pretty import pretty_repr

from scripts.data.augmentation.tacil import augmentation as tacil
from scripts.data.base.episode_store import download_episode_store
from scripts.evaluation.core.condition import (
    Condition,
    DisturbanceLevel,
    FloorCondition,
)
from scripts.evaluation.evaluate_policy import (
    DEFAULT_EPISODE_COUNT,
    STAGE_PREFIX,
    evaluate_policy,
)
from scripts.policies.dp import config as policy_config
from scripts.policies.dp.policy import load_diffusion_policy
from scripts.policies.dp.train import train_diffusion_policy
from scripts.shared.log import create_logger, log_device_info
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH
from scripts.shared.seed import set_seed
from scripts.shared.wandb_config import WandbConfig

logger = create_logger()

Augmentation = Annotated[tacil.Config, tyro.conf.subcommand(name="tacil")] | None


@dataclass(frozen=True)
class Config:
    wandb: WandbConfig

    floor_condition: FloorCondition
    disturbance_level: DisturbanceLevel | None

    expert_dataset_artifact: str

    seed: int

    augmentation: Augmentation = None


def main() -> None:
    config = tyro.cli(f=Config, config=[tyro.conf.UsePythonSyntaxForLiteralCollections])
    logger.info(pretty_repr(config))
    _run(config=config)


def _run(*, config: Config) -> None:
    condition = Condition(
        floor_condition=config.floor_condition,
        disturbance_level=config.disturbance_level,
    )
    device = torch.device(device="cuda" if torch.cuda.is_available() else "cpu")
    log_device_info(device=device, logger=logger)

    policy_training_config = policy_config.Config(
        wandb=WandbConfig(project=config.wandb.project),
        dimensionality=policy_config.DimensionalityConfig(),
        horizon_length=policy_config.HorizonLengthConfig(),
        optimization=policy_config.OptimizationConfig(),
        evaluation=policy_config.EvaluationConfig(
            floor_condition=config.floor_condition,
            disturbance_level=config.disturbance_level,
        ),
        seed=config.seed,
    )

    with (
        wandb.init(
            project=config.wandb.project,
            notes=_build_note(config=config, condition=condition),
            job_type="dp-pipeline",
            config=asdict(obj=config),
            dir=OUTPUT_DIRECTORY_PATH,
        ) as run,
        tempfile.TemporaryDirectory() as checkpoint_directory,
    ):
        expert_episodes = download_episode_store(
            artifact_name=config.expert_dataset_artifact, run=run
        )

        augmented_episodes = None
        if config.augmentation is not None:
            logger.info("Stage 1/3: creating the augmented episodes.")
            augmented_episodes = tacil.augment(
                config=config.augmentation,
                expert_episode_store=expert_episodes,
                # The expert continuation behind every synthetic state is sized
                # against this horizon.
                prediction_horizon=policy_training_config.horizon_length.prediction,
                seed=config.seed,
                device=device,
                run=run,
            )
            logger.info("Stage 1/3: done.")

        logger.info("Stage 2/3: training the diffusion policy.")
        checkpoint_path = train_diffusion_policy(
            config=policy_training_config,
            expert_episodes=expert_episodes,
            augmented_episodes=augmented_episodes,
            checkpoint_directory_path=Path(checkpoint_directory),
            device=device,
            run=run,
        )
        logger.info("Stage 2/3: done.")

        # Score the selected checkpoint the way it was selected.
        sampler = policy_training_config.evaluation.sampler
        set_seed(seed=config.seed)
        logger.info("Stage 3/3: evaluating the policy.")
        evaluate_policy(
            policy_factory=load_diffusion_policy(
                checkpoint_path=checkpoint_path, sampler=sampler, device=device
            ),
            policy_label=f"dp ({sampler})   diffusion-policy-{run.name}",
            condition=condition,
            episode_count=DEFAULT_EPISODE_COUNT,
            run=run,
        )
        logger.info("Stage 3/3: done.")

        reward_key = f"{STAGE_PREFIX}{condition.label}/reward_mean"
        logger.info("Trial finished: %s = %s", reward_key, run.summary[reward_key])


def _build_note(*, config: Config, condition: Condition) -> str:
    augmentation = config.augmentation
    augmentation_note = (
        "aug=none"
        if augmentation is None
        else (
            f"aug=tacil"
            f"|lip={augmentation.lipschitz_constant_upper_bound}"
            f"|win={augmentation.window_length}"
            f"|noise={augmentation.action_noise_fraction:.4f}"
            f"|err={augmentation.error_limit_fraction:.4f}"
            f"|dw={augmentation.disagreement_weight:.2f}"
            f"|prev={augmentation.include_previous_state}"
            f"|rel={augmentation.measure_positions_from_agent}"
        )
    )

    return (
        f"data={config.expert_dataset_artifact}"
        f"|{augmentation_note}"
        f"|seed={config.seed}"
        f"|cond={condition.label}"
    )


if __name__ == "__main__":
    main()
