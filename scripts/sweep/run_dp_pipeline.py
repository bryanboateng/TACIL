"""A diffusion policy trained on the raw expert data.

The unaugmented counterpart to run_pipeline: same policy, same evaluation,
without the dynamics model and the synthetic labels. What TaCIL has to beat
to be worth its machinery. With the policy hyperparameters fixed there is
nothing to search, so this runs once rather than as a sweep.
"""

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
from scripts.policies.dp import config as policy_config
from scripts.policies.dp.policy import load_diffusion_policy
from scripts.policies.dp.train import train_diffusion_policy
from scripts.shared.data import load_episode_store
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
        job_type="dp-pipeline",
        config=asdict(obj=config),
        dir=OUTPUT_DIRECTORY_PATH,
    ) as run:
        expert_episode_store = load_episode_store(
            episode_file_path=Path(
                run.use_artifact(
                    artifact_or_name=config.expert_dataset_artifact
                ).download()
            )
        )

        policy_training_config = policy_config.Config(
            wandb=WandbConfig(project=config.wandb.project),
            dataset_artifact=PASSED_IN_MEMORY,
            dimensionality=policy_config.DimensionalityConfig(),
            horizon_length=policy_config.HorizonLengthConfig(),
            optimization=policy_config.OptimizationConfig(),
            evaluation=policy_config.EvaluationConfig(
                floor_condition=config.floor_condition,
                disturbance_level=config.disturbance_level,
            ),
            seed=config.seed,
        )

        with tempfile.TemporaryDirectory() as checkpoint_directory:
            logger.info("Stage 1/2: training the diffusion policy.")
            checkpoint_path = train_diffusion_policy(
                config=policy_training_config,
                episode_store=expert_episode_store,
                checkpoint_directory_path=Path(checkpoint_directory),
                device=device,
                run=run,
            )
            logger.info("Stage 1/2: done.")

            # Score the selected checkpoint the way it was selected.
            sampler = policy_training_config.evaluation.sampler
            set_seed(seed=config.seed)
            logger.info("Stage 2/2: evaluating the policy.")
            evaluate_policy(
                policy_factory=load_diffusion_policy(
                    checkpoint_path=checkpoint_path, sampler=sampler, device=device
                ),
                policy_label=f"dp ({sampler})   diffusion-policy-{run.name}",
                condition=condition,
                episode_count=DEFAULT_EPISODE_COUNT,
                run=run,
            )
            logger.info("Stage 2/2: done.")

        reward_key = f"{STAGE_PREFIX}{condition.label}/reward_mean"
        logger.info("Trial finished: %s = %s", reward_key, run.summary[reward_key])


if __name__ == "__main__":
    main()
