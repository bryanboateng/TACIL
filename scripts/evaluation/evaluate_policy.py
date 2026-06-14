from dataclasses import asdict
from pathlib import Path

import torch
import tyro
import wandb
from rich.pretty import pretty_repr
from wandb import Run

from scripts.evaluation.bryan.evaluation import RewardStatistics, evaluate_policy
from scripts.evaluation.config import (
    BehaviorCloningCheckpoint,
    Config,
    DiffusionPolicyCheckpoint,
)
from scripts.policies.bc.policy import load_behavior_cloning_policy
from scripts.policies.dp.policy import load_diffusion_policy
from scripts.policies.gpi import config as gpi
from scripts.policies.gpi.policy import load_geometric_policy
from scripts.shared.data import load_episode_dataset
from scripts.shared.logging import create_logger, log_device_info
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH
from scripts.shared.policy_types import PolicyFactory
from scripts.shared.seed import set_seed

logger = create_logger()


def main():
    config = tyro.cli(
        f=Config,
        config=[
            tyro.conf.UsePythonSyntaxForLiteralCollections,
            tyro.conf.OmitSubcommandPrefixes,
        ],
    )
    logger.info(pretty_repr(config))
    _run(config=config)


def _run(*, config: Config) -> None:
    set_seed(seed=config.seed)
    device = torch.device(device="cuda" if torch.cuda.is_available() else "cpu")
    log_device_info(device=device, logger=logger)

    with wandb.init(
        project="abc",
        job_type="evaluate-policy",
        config=asdict(config),
        dir=OUTPUT_DIRECTORY_PATH,
    ) as run:
        policy_factory = _build_policy_factory(config=config, run=run, device=device)

        clean_result = evaluate_policy(
            policy_factory=policy_factory,
            episode_seeds=config.episode_seeds,
            disturber_config=None,
            description="Undisturbed",
        )
        _write_summary(run=run, protocol="clean", max_reward_statistics=clean_result)

        disturbed_result = evaluate_policy(
            policy_factory=policy_factory,
            episode_seeds=config.episode_seeds,
            disturber_config=config.disturbance,
            description="Disturbed",
        )
        _write_summary(
            run=run, protocol="disturbed", max_reward_statistics=disturbed_result
        )


def _build_policy_factory(
    *, config: Config, run: Run, device: torch.device
) -> PolicyFactory:
    match config.policy:
        case DiffusionPolicyCheckpoint(checkpoint_artifact=artifact_name):
            return load_diffusion_policy(
                checkpoint_path=_download_checkpoint(
                    run=run, artifact_name=artifact_name
                ),
                device=device,
            )
        case BehaviorCloningCheckpoint(checkpoint_artifact=artifact_name):
            return load_behavior_cloning_policy(
                checkpoint_path=_download_checkpoint(
                    run=run, artifact_name=artifact_name
                ),
                device=device,
            )
        case gpi.Config() as gpi_config:
            episode_dataset = load_episode_dataset(
                artifact=run.use_artifact(artifact_or_name=gpi_config.dataset_artifact),
                logger=logger,
            )
            return load_geometric_policy(
                episode_dataset=episode_dataset, config=gpi_config, device=device
            )


def _download_checkpoint(*, run: Run, artifact_name: str) -> Path:
    return (
        Path(run.use_artifact(artifact_or_name=artifact_name).download())
        / "checkpoint.pt"
    )


def _write_summary(
    *, run: Run, protocol: str, max_reward_statistics: RewardStatistics
) -> None:
    run.summary[f"{protocol}/max_reward_mean"] = max_reward_statistics.mean
    run.summary[f"{protocol}/max_reward_std"] = max_reward_statistics.std


if __name__ == "__main__":
    main()
