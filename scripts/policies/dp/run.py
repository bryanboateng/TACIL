from dataclasses import asdict
from typing import Annotated, TypeAlias

import torch
import tyro
import wandb
from rich.pretty import pretty_repr

from scripts.data.augmentation.tacil import augmentation as tacil
from scripts.data.augmentation.tacil.augmentation import TacilConfig
from scripts.evaluation.evaluate_policy import evaluate_policy
from scripts.policies.dp import config as policy_config
from scripts.policies.dp.policy import create_diffusion_policy_factory
from scripts.policies.dp.train import train_diffusion_policy
from scripts.policies.shared.run_config import RunConfig
from scripts.shared.dataset_artifact import download_episode_store
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
    device = torch.device(device="cuda" if torch.cuda.is_available() else "cpu")
    log_device_info(device=device, logger=logger)

    with wandb.init(
        project=config.wandb_project,
        job_type="dp",
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

        match config.augmentation:
            case None:
                augmented_episodes = None
                logger.info("Stage 1/3: skipped, no augmentation configured.")
            case TacilConfig() as tacil_configuration:
                augmented_episodes = tacil.augment(
                    config=tacil_configuration,
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
            expert_episodes=expert_episodes,
            augmented_episodes=augmented_episodes,
            seed=config.seed,
            device=device,
            run=run,
        )
        logger.info("Stage 2/3: done.")

        logger.info("Stage 3/3: evaluating the policy.")
        evaluate_policy(
            policy_factory=create_diffusion_policy_factory(
                noise_prediction_network=noise_prediction_network,
                horizon_lengths=config.policy.horizon_lengths,
                noise_scheduler_name=config.policy.sampler,
                device=device,
            ),
            policy_label=f"dp ({run_name})",
            seed=config.seed,
            record_videos=config.record_videos,
            run=run,
        )
        logger.info("Stage 3/3: done.")


if __name__ == "__main__":
    main()
