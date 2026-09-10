from dataclasses import asdict
from typing import TypeAlias

import torch
import tyro
import wandb
from rich.pretty import pretty_repr

from scripts.environments.registry import create_environment
from scripts.evaluation.evaluate_policy import evaluate_policy
from scripts.policies.gpi import config as gpi_config
from scripts.policies.gpi.policy import create_geometric_policy_factory
from scripts.policies.shared.run_config import RunConfig
from scripts.shared import stage_names
from scripts.shared.episode_store import download_episode_store
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

    environment = create_environment(environment=config.environment)

    with wandb.init(
        project=config.wandb_project,
        job_type="gpi",
        config=asdict(obj=config),
        dir=OUTPUT_DIRECTORY_PATH,
    ) as run:
        run_name = run.name
        if run_name is None:
            raise RuntimeError("wandb run name unexpectedly None")

        expert_episodes = download_episode_store(
            artifact_name=config.expert_dataset_artifact, run=run
        )

        run.summary[f"{stage_names.POLICY}/0_dataset/1_transition_count"] = len(
            expert_episodes.states
        )

        logger.info("Stage 1/1: evaluating the geometric policy.")
        evaluate_policy(
            environment=environment,
            policy_factory=create_geometric_policy_factory(
                episode_store=expert_episodes,
                geometry=environment.gpi_geometry,
                config=config.policy,
                device=device,
            ),
            policy_label=f"gpi ({run_name})",
            seed=config.seed,
            run=run,
        )
        logger.info("Stage 1/1: done.")


if __name__ == "__main__":
    main()
