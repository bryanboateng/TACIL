"""Standalone dynamics stage: train the ensemble, run the diagnostics, stop.

For checking the dynamics model and its error signals without paying for the
policy stages.
"""

from dataclasses import asdict, dataclass, field

import torch
import tyro
import wandb
from rich.pretty import pretty_repr

from scripts.data.augmentation.tacil.create_dynamics.config import DynamicsConfig
from scripts.data.augmentation.tacil.create_dynamics.dynamics_model_ensemble_training import (  # noqa: E501
    train_dynamics_model_ensemble,
)
from scripts.shared.episode_store import download_episode_store
from scripts.shared.log import create_logger, log_device_info
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH

logger = create_logger()


@dataclass(frozen=True)
class Config:
    seed: int
    wandb_project: str
    expert_dataset_artifact: str
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)


def main() -> None:
    config = tyro.cli(f=Config, config=[tyro.conf.UsePythonSyntaxForLiteralCollections])
    logger.info(pretty_repr(config))

    device = torch.device(device="cuda" if torch.cuda.is_available() else "cpu")
    log_device_info(device=device, logger=logger)

    with wandb.init(
        project=config.wandb_project,
        job_type="dynamics",
        config=asdict(obj=config),
        dir=OUTPUT_DIRECTORY_PATH,
    ) as run:
        expert_episodes = download_episode_store(
            artifact_name=config.expert_dataset_artifact, run=run
        )
        train_dynamics_model_ensemble(
            config=config.dynamics,
            episode_store=expert_episodes,
            seed=config.seed,
            device=device,
            run=run,
        )


if __name__ == "__main__":
    main()
