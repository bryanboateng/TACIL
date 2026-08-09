from dataclasses import asdict
from typing import Annotated, TypeAlias

import numpy as np
import torch
import tyro
import wandb
from rich.pretty import pretty_repr
from wandb import Run

from scripts.data.augmentation.ccil import augmentation as ccil
from scripts.data.augmentation.ccil.config import CcilConfig
from scripts.data.augmentation.tacil import augmentation as tacil
from scripts.data.augmentation.tacil.augmentation import TacilConfig
from scripts.evaluation.core.condition import Condition
from scripts.evaluation.evaluate_policy import evaluate_policy
from scripts.policies.bc import config as bc_config
from scripts.policies.bc.policy import create_behavior_cloning_policy_factory
from scripts.policies.bc.train import train_behavior_cloning_policy
from scripts.policies.shared.run_config import RunConfig
from scripts.shared.f import EpisodeStore, TransitionBatch, download_episode_store
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
        run_name = run.name
        if run_name is None:
            raise RuntimeError("wandb run name unexpectedly None")

        expert_episodes = download_episode_store(
            artifact_name=config.expert_dataset_artifact, run=run
        )

        logger.info("Stage 1/3: building the training transitions.")
        expert_transitions = TransitionBatch(
            states=expert_episodes.states, actions=expert_episodes.actions
        )
        if config.augmentation is None:
            transitions = expert_transitions
        else:
            synthetic_transitions = _create_synthetic_transitions(
                augmentation_config=config.augmentation,
                expert_episodes=expert_episodes,
                seed=config.seed,
                device=device,
                run=run,
            )
            transitions = TransitionBatch(
                states=np.concatenate(
                    [expert_transitions.states, synthetic_transitions.states], axis=0
                ),  # (transition_count, state_dimensionality)
                actions=np.concatenate(
                    [expert_transitions.actions, synthetic_transitions.actions], axis=0
                ),  # (transition_count, action_dimensionality)
            )

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
            policy_factory=create_behavior_cloning_policy_factory(
                policy_model=policy_model, device=device
            ),
            policy_label=f"bc ({run_name})",
            condition=condition,
            run=run,
        )
        logger.info("Stage 3/3: done.")


def _create_synthetic_transitions(
    *,
    augmentation_config: Augmentation,
    expert_episodes: EpisodeStore,
    seed: int,
    device: torch.device,
    run: Run,
) -> TransitionBatch:
    match augmentation_config:
        case CcilConfig() as ccil_configuration:
            return ccil.augment(
                config=ccil_configuration,
                expert_episodes=expert_episodes,
                seed=seed,
                device=device,
                run=run,
            )
        case TacilConfig() as tacil_configuration:
            tacil_episodes = tacil.augment(
                config=tacil_configuration,
                expert_episodes=expert_episodes,
                prediction_horizon=None,  # Behavior cloning trains on single (state, action) pairs.
                seed=seed,
                device=device,
                run=run,
            )
            return TransitionBatch(
                states=tacil_episodes.states, actions=tacil_episodes.actions
            )


if __name__ == "__main__":
    main()
