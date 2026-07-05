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
)
from scripts.evaluation.core.condition import Condition, get_conditions
from scripts.evaluation.core.scoring import (
    RewardStatistics,
    pick_representative_scores,
    reward_statistics,
    run_scored_episodes,
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
        project="abc",
        job_type="evaluate-policy",
        config=asdict(config),
        dir=OUTPUT_DIRECTORY_PATH,
    ) as run:
        start_episode_seed = 1_000_000
        policy_factory = _build_policy_factory(config=config, run=run, device=device)

        for condition in get_conditions():
            episode_scores = run_scored_episodes(
                policy_factory=policy_factory,
                start_episode_seed=start_episode_seed,
                episode_count=config.episode_count,
                condition=condition,
            )
            record_condition(
                policy_factory=policy_factory,
                condition=condition,
                representative_scores=pick_representative_scores(
                    episode_scores=episode_scores
                ),
                run=run,
            )
            _write_summary(
                run=run, condition=condition, reward=reward_statistics(episode_scores)
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


def _write_summary(*, run: Run, condition: Condition, reward: RewardStatistics) -> None:
    run.summary[f"{condition.label}/reward_mean"] = reward.mean
    run.summary[f"{condition.label}/reward_std"] = reward.std


if __name__ == "__main__":
    main()
