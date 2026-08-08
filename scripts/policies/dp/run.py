"""Diffusion policy, optionally on TaCIL-augmented data: one trial end to end.

The augmentation runs inside this process and its output never becomes an
artifact. Every trial searches parameters that change that output, so there
would be nothing to reuse it for.

CCIL is not offered here: its labels are single (state, action) pairs, which
cannot fill a window.
"""

from dataclasses import asdict
from typing import Annotated, TypeAlias

import torch
import tyro
import wandb
from rich.pretty import pretty_repr

from scripts.data.augmentation.tacil import augmentation as tacil
from scripts.data.augmentation.tacil.augmentation import TacilConfig
from scripts.evaluation.core.condition import Condition
from scripts.evaluation.evaluate_policy import (
    DEFAULT_EPISODE_COUNT,
    STAGE_NAME,
    evaluate_policy,
)
from scripts.policies.dp import config as policy_config
from scripts.policies.dp.train import build_policy_factory, train_diffusion_policy
from scripts.policies.shared.run_config import RunConfig
from scripts.shared.f import download_episode_store
from scripts.shared.log import create_logger, log_device_info
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH

logger = create_logger()

Augmentation: TypeAlias = Annotated[TacilConfig, tyro.conf.subcommand(name="tacil")]
Config: TypeAlias = RunConfig[policy_config.Config, Augmentation]


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

    with wandb.init(
        project=config.wandb_project,
        job_type="dp-pipeline",
        config=asdict(obj=config),
        dir=OUTPUT_DIRECTORY_PATH,
    ) as run:
        run_name = run.name
        if run_name is None:
            raise RuntimeError("wandb run name unexpectedly None")

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
                prediction_horizon=config.policy.horizon_length.prediction,
                seed=config.seed,
                device=device,
                run=run,
            )
            logger.info("Stage 1/3: done.")

        logger.info("Stage 2/3: training the diffusion policy.")
        noise_prediction_network = train_diffusion_policy(
            config=config.policy,
            expert_episodes=expert_episodes,
            augmented_episodes=augmented_episodes,
            condition=condition,
            seed=config.seed,
            device=device,
            run=run,
        )
        logger.info("Stage 2/3: done.")

        logger.info("Stage 3/3: evaluating the policy.")
        evaluate_policy(
            policy_factory=build_policy_factory(
                config=config.policy,
                noise_prediction_network=noise_prediction_network,
                device=device,
            ),
            policy_label=f"dp ({run_name})",
            condition=condition,
            episode_count=DEFAULT_EPISODE_COUNT,
            run=run,
        )
        logger.info("Stage 3/3: done.")

        reward_key = f"{STAGE_NAME}/{condition.label}/reward_mean"
        logger.info("Trial finished: %s = %s", reward_key, run.summary[reward_key])


if __name__ == "__main__":
    main()
