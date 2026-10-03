from typing import TypeAlias

import torch
import tyro
import wandb
from rich.pretty import pretty_repr

from scripts.environments.registry import create_environment
from scripts.evaluation.evaluate_policy import evaluate_policy
from scripts.policies.gpi import config as gpi_config
from scripts.policies.gpi.policy import create_geometric_policy_factory
from scripts.policies.shared.run_config import RunConfig, build_run_record
from scripts.shared import stage_names
from scripts.shared.dataset_artifact import download_episodes
from scripts.shared.episodes import drop_rewards
from scripts.shared.log import create_logger, log_device_info
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH
from scripts.shared.seeding import set_seed

logger = create_logger()

Config: TypeAlias = RunConfig[gpi_config.Config, None]


def main() -> None:
    config = tyro.cli(f=Config)
    logger.info(pretty_repr(config))
    _run(config=config)


def _run(*, config: Config) -> None:
    set_seed(seed=config.seed)

    device = torch.device(device="cuda" if torch.cuda.is_available() else "cpu")
    log_device_info(device=device, logger=logger)

    with wandb.init(
        project=config.wandb_project,
        job_type="gpi",
        config=build_run_record(
            config=config, policy_name="gpi", augmentation_name=None
        ),
        dir=OUTPUT_DIRECTORY_PATH,
    ) as run:
        environment = create_environment(name=config.environment)

        dataset = download_episodes(
            artifact_name=config.expert_dataset_artifact, run=run
        )
        run.config["expert_dataset_artifact_version"] = dataset.artifact_version
        expert_episodes = drop_rewards(episodes=dataset.episodes)

        run.summary[f"{stage_names.POLICY}/0_dataset/1_transition_count"] = len(
            expert_episodes.states
        )

        logger.info("Stage 1/1: evaluating the geometric policy.")
        evaluate_policy(
            environment=environment,
            policy_factory=create_geometric_policy_factory(
                gpi_geometry=environment.gpi_geometry,
                episodes=expert_episodes,
                config=config.policy,
                device=device,
            ),
            config=config.evaluation,
            run=run,
        )
        logger.info("Stage 1/1: done.")


if __name__ == "__main__":
    main()
