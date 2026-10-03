from typing import TypeAlias

import torch
import tyro
import wandb
from rich.pretty import pretty_repr

from scripts.data.augmentation.augment import (
    AugmentationConfig,
    augment,
    get_augmentation_name,
)
from scripts.environments.registry import create_environment
from scripts.evaluation.evaluate_policy import evaluate_policy
from scripts.policies.dp import config as policy_config
from scripts.policies.dp.policy import create_diffusion_policy_factory
from scripts.policies.dp.train import train_diffusion_policy
from scripts.policies.shared.run_config import RunConfig, build_run_record
from scripts.policies.shared.validation import Validation
from scripts.shared.dataset_artifact import download_episodes
from scripts.shared.episodes import drop_rewards
from scripts.shared.log import create_logger, log_device_info
from scripts.shared.model_artifact import POLICY_ARTIFACT_TYPE, upload_model
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH

logger = create_logger()

Config: TypeAlias = RunConfig[policy_config.Config, AugmentationConfig]


def main() -> None:
    config = tyro.cli(f=Config, config=[tyro.conf.UsePythonSyntaxForLiteralCollections])
    logger.info(pretty_repr(config))
    _run(config=config)


def _run(*, config: Config) -> None:
    device = torch.device(device="cuda" if torch.cuda.is_available() else "cpu")
    log_device_info(device=device, logger=logger)

    with wandb.init(
        project=config.wandb_project,
        job_type="dp",
        config=build_run_record(
            config=config,
            policy_name="dp",
            augmentation_name=get_augmentation_name(config=config.augmentation),
        ),
        dir=OUTPUT_DIRECTORY_PATH,
    ) as run:
        environment = create_environment(name=config.environment)

        dataset = download_episodes(
            artifact_name=config.expert_dataset_artifact, run=run
        )
        run.config["expert_dataset_artifact_version"] = dataset.artifact_version
        expert_episodes = drop_rewards(episodes=dataset.episodes)

        logger.info("Stage 1/3: creating the augmented episodes.")

        if config.augmentation is None:
            augmented_episodes = None
            logger.info("Stage 1/3: skipped, no augmentation configured.")
        else:
            augmented_episodes = augment(
                config=config.augmentation,
                environment=environment,
                expert_episodes=expert_episodes,
                # The policy trains on windows of `prediction` steps. The
                # expert steps let the last windows run from synthetic
                # states into recorded ones, which is what teaches the
                # policy how to get back on track.
                #
                #     step        0 1 2 3 4 5 6 7
                #     synthetic   s s s
                #     expert            e e e e e     prediction - 1
                #     window 0    w w w w w w
                #     window 1      w w w w w w
                #     window 2        w w w w w w
                #
                # One step fewer than a full window keeps every window
                # starting on a synthetic state. More would add windows
                # that hold expert steps only, and those just repeat data
                # the policy already has.
                expert_part_length=config.policy.horizon_lengths.prediction - 1,
                seed=config.seed,
                device=device,
                run=run,
            )
        logger.info("Stage 1/3: done.")

        logger.info("Stage 2/3: training the diffusion policy.")
        noise_prediction_network = train_diffusion_policy(
            config=config.policy,
            validation=Validation(
                settings=config.policy.validation, environment=environment
            ),
            expert_episodes=expert_episodes,
            augmented_episodes=augmented_episodes,
            seed=config.seed,
            device=device,
            run=run,
        )
        if config.policy.save_model:
            upload_model(
                model=noise_prediction_network,
                artifact_type=POLICY_ARTIFACT_TYPE,
                run=run,
            )
        logger.info("Stage 2/3: done.")

        logger.info("Stage 3/3: evaluating the policy.")
        evaluate_policy(
            environment=environment,
            policy_factory=create_diffusion_policy_factory(
                noise_prediction_network=noise_prediction_network,
                horizon_lengths=config.policy.horizon_lengths,
                noise_scheduler_name=config.policy.sampler,
                device=device,
            ),
            config=config.evaluation,
            run=run,
        )
        logger.info("Stage 3/3: done.")


if __name__ == "__main__":
    main()
