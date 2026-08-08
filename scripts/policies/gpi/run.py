"""Pipeline for one sweep trial: GPI (geometric policy) — no training, just
evaluation of the retrieval policy under the condition."""

from dataclasses import asdict

import torch
import tyro
import wandb
from rich.pretty import pretty_repr

from scripts.evaluation.core.condition import Condition
from scripts.evaluation.evaluate_policy import (
    DEFAULT_EPISODE_COUNT,
    STAGE_PREFIX,
    evaluate_policy,
)
from scripts.policies.gpi import config as gpi_config
from scripts.policies.gpi.policy import load_geometric_policy
from scripts.policies.shared.run_config import RunConfig
from scripts.data.base.episode_store import download_episode_store
from scripts.shared.log import create_logger, log_device_info
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH
from scripts.shared.seeding import set_seed

logger = create_logger()


Config = RunConfig[gpi_config.Config]


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
        project=config.wandb.project,
        notes=(
            f"data={config.expert_dataset_artifact}"
            f"|l1={config.policy.guidance.lambda1:.3f}"
            f"|ah={config.policy.action_horizon}"
            f"|seed={config.seed}"
            f"|cond={condition.label}"
        ),
        job_type="gpi-pipeline",
        config=asdict(obj=config),
        dir=OUTPUT_DIRECTORY_PATH,
    ) as run:
        episode_store = download_episode_store(
            artifact_name=config.expert_dataset_artifact, run=run
        )

        set_seed(seed=config.seed)
        logger.info("Evaluating the geometric policy.")
        evaluate_policy(
            policy_factory=load_geometric_policy(
                episode_store=episode_store, config=config.policy, device=device
            ),
            policy_label=f"gpi   {config.expert_dataset_artifact}",
            condition=condition,
            episode_count=DEFAULT_EPISODE_COUNT,
            run=run,
        )

        reward_key = f"{STAGE_PREFIX}{condition.label}/reward_mean"
        logger.info("Trial finished: %s = %s", reward_key, run.summary[reward_key])


if __name__ == "__main__":
    main()
