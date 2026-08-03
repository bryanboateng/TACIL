from dataclasses import asdict
from pathlib import Path

import torch
import tyro
import wandb
from rich.pretty import pretty_repr
from wandb import Run

from scripts.evaluation.config import (
    BehaviorCloningCheckpoint,
    Config,
    DiffusionPolicyCheckpoint,
    Policy,
)
from scripts.evaluation.core.condition import Condition
from scripts.evaluation.core.scoring import (
    RewardStatistics,
    pick_representative_episodes,
    reward_statistics,
    run_scored_episodes,
    success_rate,
)
from scripts.evaluation.core.video import record_condition
from scripts.policies.bc.policy import load_behavior_cloning_policy
from scripts.policies.dp.policy import load_diffusion_policy
from scripts.policies.gpi import config as gpi
from scripts.policies.gpi.policy import load_geometric_policy
from scripts.shared.data import load_episode_store
from scripts.shared.log import create_logger, log_device_info
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
        project=config.wandb.project,
        notes=config.wandb.notes,
        job_type="evaluate-policy",
        config=asdict(config),
        dir=OUTPUT_DIRECTORY_PATH,
    ) as run:
        evaluate_policy(
            policy_factory=(
                _build_policy_factory(config=config, run=run, device=device)
            ),
            policy_label=_describe_policy(policy=config.policy),
            condition=Condition(
                floor_condition=config.floor_condition,
                disturbance_level=config.disturbance_level,
            ),
            episode_count=config.episode_count,
            run=run,
        )


def evaluate_policy(
    *,
    policy_factory: PolicyFactory,
    policy_label: str,
    condition: Condition,
    episode_count: int,
    run: Run,
) -> None:
    episode_records = run_scored_episodes(
        policy_factory=policy_factory,
        start_episode_seed=1_000_000,
        episode_count=episode_count,
        condition=condition,
    )

    _write_summary(
        run=run,
        condition=condition,
        reward=reward_statistics(episode_records),
        success_fraction=success_rate(episode_records),
    )
    record_condition(
        policy_label=policy_label,
        condition=condition,
        episode_records=pick_representative_episodes(episode_records=episode_records),
        run=run,
    )


def _build_policy_factory(
    *, config: Config, run: Run, device: torch.device
) -> PolicyFactory:
    match config.policy:
        case DiffusionPolicyCheckpoint(
            checkpoint_artifact=artifact_name, sampler=sampler
        ):
            return load_diffusion_policy(
                checkpoint_path=_download_checkpoint(
                    run=run, artifact_name=artifact_name
                ),
                sampler=sampler,
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
            episode_store = load_episode_store(
                episode_file_path=Path(
                    run.use_artifact(
                        artifact_or_name=gpi_config.dataset_artifact
                    ).download()
                )
            )
            return load_geometric_policy(
                episode_store=episode_store, config=gpi_config, device=device
            )


def _download_checkpoint(*, run: Run, artifact_name: str) -> Path:
    return (
        Path(run.use_artifact(artifact_or_name=artifact_name).download())
        / "checkpoint.pt"
    )


def _describe_policy(*, policy: Policy) -> str:
    match policy:
        case DiffusionPolicyCheckpoint(
            checkpoint_artifact=artifact_name, sampler=sampler
        ):
            return f"dp ({sampler})   {artifact_name}"
        case BehaviorCloningCheckpoint(checkpoint_artifact=artifact_name):
            return f"bc   {artifact_name}"
        case gpi.Config(dataset_artifact=artifact_name):
            return f"gpi   {artifact_name}"


def _write_summary(
    *, run: Run, condition: Condition, reward: RewardStatistics, success_fraction: float
) -> None:
    run.summary[f"{condition.label}/reward_mean"] = reward.mean
    run.summary[f"{condition.label}/reward_std"] = reward.std
    run.summary[f"{condition.label}/success_rate"] = success_fraction


if __name__ == "__main__":
    main()
