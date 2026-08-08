"""Behavior cloning, optionally on augmented data: one sweep trial end to end.

The augmentation runs inside this process and its output never becomes an
artifact. Every trial searches parameters that change that output, so there
would be nothing to reuse it for.
"""

import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, TypeAlias

import torch
import tyro
import wandb
from rich.pretty import pretty_repr
from wandb import Run

from scripts.data.augmentation.ccil import augmentation as ccil
from scripts.data.augmentation.ccil.config import CcilConfig
from scripts.data.augmentation.ccil.f import (
    TransitionBatch,
    concatenate_transitions,
    to_transitions,
)
from scripts.data.augmentation.tacil import augmentation as tacil
from scripts.data.augmentation.tacil.augmentation import TacilConfig
from scripts.data.base.episode_store import EpisodeStore, download_episode_store
from scripts.evaluation.core.condition import Condition
from scripts.evaluation.evaluate_policy import (
    DEFAULT_EPISODE_COUNT,
    STAGE_PREFIX,
    evaluate_policy,
)
from scripts.policies.bc import config as bc_config
from scripts.policies.bc.policy import load_behavior_cloning_policy
from scripts.policies.bc.train import train_behavior_cloning_policy
from scripts.policies.shared.run_config import RunConfig
from scripts.shared.log import create_logger, log_device_info
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH
from scripts.shared.seeding import set_seed

logger = create_logger()

Augmentation: TypeAlias = (
    Annotated[TacilConfig, tyro.conf.subcommand(name="tacil")]
    | Annotated[CcilConfig, tyro.conf.subcommand(name="ccil")]
)
Config: TypeAlias = RunConfig[bc_config.Config, Augmentation]


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

    with (
        wandb.init(
            project=config.wandb.project,
            notes=_build_note(config=config, condition=condition),
            job_type="bc-pipeline",
            config=asdict(obj=config),
            dir=OUTPUT_DIRECTORY_PATH,
        ) as run,
        tempfile.TemporaryDirectory() as checkpoint_directory,
    ):
        expert_episodes = download_episode_store(
            artifact_name=config.expert_dataset_artifact, run=run
        )

        logger.info("Stage 1/3: building the training transitions.")
        transitions = _build_transitions(
            config=config, expert_episodes=expert_episodes, device=device, run=run
        )
        run.summary["transition_count"] = len(transitions.states)
        logger.info("Stage 1/3: done.")

        logger.info("Stage 2/3: training the behavior-cloning policy.")
        checkpoint_path = train_behavior_cloning_policy(
            config=config.policy,
            transitions=transitions,
            condition=condition,
            seed=config.seed,
            checkpoint_directory_path=Path(checkpoint_directory),
            device=device,
            run=run,
        )
        logger.info("Stage 2/3: done.")

        set_seed(seed=config.seed)
        logger.info("Stage 3/3: evaluating the policy.")
        evaluate_policy(
            policy_factory=load_behavior_cloning_policy(
                checkpoint_path=checkpoint_path, device=device
            ),
            policy_label=f"bc   behavior-cloning-policy-{run.name!s}",
            condition=condition,
            episode_count=DEFAULT_EPISODE_COUNT,
            run=run,
        )
        logger.info("Stage 3/3: done.")

        reward_key = f"{STAGE_PREFIX}{condition.label}/reward_mean"
        logger.info("Trial finished: %s = %s", reward_key, run.summary[reward_key])


def _build_transitions(
    *, config: Config, expert_episodes: EpisodeStore, device: torch.device, run: Run
) -> TransitionBatch:
    expert_transitions = to_transitions(episode_store=expert_episodes)
    augmentation = config.augmentation

    match augmentation:
        case None:
            return expert_transitions
        case CcilConfig() as ccil_configuration:
            label_transitions = ccil.augment(
                config=ccil_configuration,
                expert_episode_store=expert_episodes,
                seed=config.seed,
                device=device,
                run=run,
            )
            return concatenate_transitions(
                transition_batches=[expert_transitions, label_transitions]
            )
        case TacilConfig() as tacil_configuration:
            augmented_episodes = tacil.augment(
                config=tacil_configuration,
                expert_episode_store=expert_episodes,
                # Behavior cloning trains on single (state, action) pairs.
                prediction_horizon=None,
                seed=config.seed,
                device=device,
                run=run,
            )
            if augmented_episodes is None:
                return expert_transitions
            return concatenate_transitions(
                transition_batches=[
                    expert_transitions,
                    to_transitions(episode_store=augmented_episodes),
                ]
            )


def _augmentation_note(*, augmentation: Augmentation | None) -> str:
    match augmentation:
        case None:
            return "aug=none"
        case CcilConfig() as ccil_configuration:
            return (
                f"aug=ccil"
                f"|ltype={ccil_configuration.dynamics.lipschitz_type}"
                f"|lip={ccil_configuration.dynamics.lipschitz_constraint:.3f}"
                f"|type={ccil_configuration.labels.type}"
                f"|anoise={ccil_configuration.labels.action_noise_std:.5f}"
                f"|q={ccil_configuration.labels.rejection_quantile:.3f}"
            )
        case TacilConfig() as tacil_configuration:
            return (
                f"aug=tacil"
                f"|lip={tacil_configuration.dynamics.lipschitz_constant_upper_bound!s}"
                f"|win={tacil_configuration.dynamics.window_length}"
                f"|noise={tacil_configuration.data.action_noise_fraction:.4f}"
                f"|err={tacil_configuration.data.error_limit_fraction:.4f}"
                f"|dw={tacil_configuration.data.disagreement_weight:.2f}"
            )


def _build_note(*, config: Config, condition: Condition) -> str:
    return (
        f"data={config.expert_dataset_artifact}"
        f"|{_augmentation_note(augmentation=config.augmentation)}"
        f"|seed={config.seed}"
        f"|cond={condition.label}"
    )


if __name__ == "__main__":
    main()
