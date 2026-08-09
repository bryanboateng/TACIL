from dataclasses import asdict
from typing import Annotated, TypeAlias

import torch
import tyro
import wandb
from rich.pretty import pretty_repr

from scripts.data.augmentation.tacil import augmentation as tacil
from scripts.data.augmentation.tacil.augmentation import TacilConfig
from scripts.evaluation.core.condition import Condition
from scripts.evaluation.evaluate_policy import evaluate_policy
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

        logger.info("Stage 1/3: creating the augmented episodes.")
        if config.augmentation is None:
            augmented_episodes = None
            logger.info("Stage 1/3: skipped, no augmentation configured.")
        else:
            augmented_episodes = tacil.augment(
                config=config.augmentation,
                expert_episodes=expert_episodes,
                # The policy trains on windows of `prediction` steps. The expert
                # steps let the last windows run from synthetic states into
                # recorded ones, which is what teaches the policy how to get back
                # on track. One step fewer than a full window keeps every window
                # starting on a synthetic state. More would add windows that hold
                # expert steps only, and those just repeat data the policy already has.
                expert_part_length=config.policy.horizon_length.prediction - 1,
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
                horizon_length=config.policy.horizon_length,
                sampler=config.policy.sampler,
                noise_prediction_network=noise_prediction_network,
                device=device,
            ),
            policy_label=f"dp ({run_name})",
            condition=condition,
            run=run,
        )
        logger.info("Stage 3/3: done.")


if __name__ == "__main__":
    main()
