"""Behavior cloning, optionally on augmented data: one sweep trial end to end.

The augmentation runs inside this process and its output never becomes an
artifact. Every trial searches parameters that change that output, so there
would be nothing to reuse it for.
"""

from dataclasses import asdict
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
from scripts.policies.bc.policy import build_behavior_cloning_policy_factory
from scripts.policies.bc.train import train_behavior_cloning_policy
from scripts.policies.shared.run_config import RunConfig
from scripts.shared.log import create_logger, log_device_info
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH

logger = create_logger()

Augmentation: TypeAlias = (
    Annotated[TacilConfig, tyro.conf.subcommand(name="tacil")]
    | Annotated[CcilConfig, tyro.conf.subcommand(name="ccil")]
)
Config: TypeAlias = RunConfig[bc_config.Config, Augmentation]


def main() -> None:
    config = tyro.cli(f=Config)
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
        project=config.wandb_project,
        job_type="bc-pipeline",
        config=asdict(obj=config),
        dir=OUTPUT_DIRECTORY_PATH,
    ) as run:
        expert_episodes = download_episode_store(
            artifact_name=config.expert_dataset_artifact, run=run
        )

        logger.info("Stage 1/3: building the training transitions.")
        transitions = _build_transitions(
            config=config, expert_episode_store=expert_episodes, device=device, run=run
        )
        run.summary["transition_count"] = len(transitions.states)
        logger.info("Stage 1/3: done.")

        logger.info("Stage 2/3: training the behavior-cloning policy.")
        policy_model = train_behavior_cloning_policy(
            config=config.policy,
            transitions=transitions,
            condition=condition,
            seed=config.seed,
            device=device,
            run=run,
        )
        logger.info("Stage 2/3: done.")

        logger.info("Stage 3/3: evaluating the policy.")
        evaluate_policy(
            policy_factory=build_behavior_cloning_policy_factory(
                policy_model=policy_model, device=device
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
    *,
    config: Config,
    expert_episode_store: EpisodeStore,
    device: torch.device,
    run: Run,
) -> TransitionBatch:
    expert_transitions = to_transitions(episode_store=expert_episode_store)

    match config.augmentation:
        case None:
            return expert_transitions
        case CcilConfig() as ccil_configuration:
            label_transitions = ccil.augment(
                config=ccil_configuration,
                expert_episode_store=expert_episode_store,
                seed=config.seed,
                device=device,
                run=run,
            )
            return concatenate_transitions(
                transition_batches=[expert_transitions, label_transitions]
            )
        case TacilConfig() as tacil_configuration:
            augmented_episode_store = tacil.augment(
                config=tacil_configuration,
                expert_episode_store=expert_episode_store,
                prediction_horizon=None,  # Behavior cloning trains on single (state, action) pairs.
                seed=config.seed,
                device=device,
                run=run,
            )
            return concatenate_transitions(
                transition_batches=[
                    expert_transitions,
                    to_transitions(episode_store=augmented_episode_store),
                ]
            )


if __name__ == "__main__":
    main()
