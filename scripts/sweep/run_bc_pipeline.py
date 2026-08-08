"""Plain behavior cloning on the raw expert data.

The baseline every augmentation has to beat. With the policy hyperparameters
fixed there is nothing to search, so this runs once rather than as a sweep."""

import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro
import wandb
from rich.pretty import pretty_repr

from scripts.evaluation.config import DEFAULT_EPISODE_COUNT
from scripts.evaluation.core.condition import (
    Condition,
    DisturbanceLevel,
    FloorCondition,
)
from scripts.evaluation.evaluate_policy import STAGE_PREFIX, evaluate_policy
from scripts.policies.bc import config as bc_config
from scripts.policies.bc.policy import load_behavior_cloning_policy
from scripts.policies.bc.train import train_behavior_cloning_policy
from scripts.shared.data import download_episode_store
from scripts.shared.log import create_logger, log_device_info
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH
from scripts.shared.seed import set_seed
from scripts.shared.wandb_config import WandbConfig
from scripts.sweep.run_pipeline import PASSED_IN_MEMORY

logger = create_logger()


@dataclass(frozen=True)
class Config:
    wandb: WandbConfig

    floor_condition: FloorCondition
    disturbance_level: DisturbanceLevel | None

    expert_dataset_artifact: str

    seed: int = 42


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
            f"|seed={config.seed}"
            f"|cond={condition.label}"
        ),
        job_type="bc-pipeline",
        config=asdict(obj=config),
        dir=OUTPUT_DIRECTORY_PATH,
    ) as run:
        expert_episode_store = download_episode_store(
            artifact_name=config.expert_dataset_artifact, run=run
        )

        with tempfile.TemporaryDirectory() as checkpoint_directory:
            logger.info("Stage 1/2: training the behavior-cloning policy.")
            checkpoint_path = train_behavior_cloning_policy(
                config=bc_config.Config(
                    wandb=WandbConfig(project=config.wandb.project),
                    dimensionality=bc_config.DimensionalityConfig(),
                    optimization=bc_config.OptimizationConfig(),
                    evaluation=bc_config.EvaluationConfig(
                        floor_condition=config.floor_condition,
                        disturbance_level=config.disturbance_level,
                    ),
                    dataset_artifact=PASSED_IN_MEMORY,
                    seed=config.seed,
                ),
                episode_store=expert_episode_store,
                checkpoint_directory_path=Path(checkpoint_directory),
                device=device,
                run=run,
            )
            logger.info("Stage 1/2: done.")

            set_seed(seed=config.seed)
            logger.info("Stage 2/2: evaluating the policy.")
            evaluate_policy(
                policy_factory=load_behavior_cloning_policy(
                    checkpoint_path=checkpoint_path, device=device
                ),
                policy_label=f"bc   behavior-cloning-policy-{run.name}",
                condition=condition,
                episode_count=DEFAULT_EPISODE_COUNT,
                run=run,
            )
            logger.info("Stage 2/2: done.")

        reward_key = f"{STAGE_PREFIX}{condition.label}/reward_mean"
        logger.info("Trial finished: %s = %s", reward_key, run.summary[reward_key])


if __name__ == "__main__":
    main()
