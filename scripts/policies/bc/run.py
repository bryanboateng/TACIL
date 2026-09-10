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
from scripts.environments.registry import create_environment
from scripts.environments.state_space import StateSpace
from scripts.evaluation.evaluate_policy import evaluate_policy
from scripts.policies.bc import config as bc_config
from scripts.policies.bc.policy import create_behavior_cloning_policy_factory
from scripts.policies.bc.train import train_behavior_cloning_policy
from scripts.policies.shared.run_config import RunConfig
from scripts.shared.episode_store import EpisodeStore, download_episode_store
from scripts.shared.log import create_logger, log_device_info
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH
from scripts.shared.state_action_pairs import StateActionPairs

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
    device = torch.device(device="cuda" if torch.cuda.is_available() else "cpu")
    log_device_info(device=device, logger=logger)

    environment = create_environment(environment=config.environment)

    with wandb.init(
        project=config.wandb_project,
        job_type="bc",
        config=asdict(obj=config),
        dir=OUTPUT_DIRECTORY_PATH,
    ) as run:
        run_name = run.name
        if run_name is None:
            raise RuntimeError("wandb run name unexpectedly None")

        expert_episodes = download_episode_store(
            artifact_name=config.expert_dataset_artifact, run=run
        )

        logger.info("Stage 1/3: building the training data.")
        expert_state_action_pairs = StateActionPairs(
            states=expert_episodes.states, actions=expert_episodes.actions
        )
        if config.augmentation is None:
            state_action_pairs = expert_state_action_pairs
        else:
            synthetic_state_action_pairs = _create_synthetic_state_action_pairs(
                state_space=environment.state_space,
                augmentation_config=config.augmentation,
                expert_episodes=expert_episodes,
                seed=config.seed,
                device=device,
                run=run,
            )
            state_action_pairs = StateActionPairs(
                states=np.concatenate(
                    [
                        expert_state_action_pairs.states,
                        synthetic_state_action_pairs.states,
                    ],
                    axis=0,
                ),  # (pair_count, state_dimensionality)
                actions=np.concatenate(
                    [
                        expert_state_action_pairs.actions,
                        synthetic_state_action_pairs.actions,
                    ],
                    axis=0,
                ),  # (pair_count, action_dimensionality)
            )

        logger.info("Stage 1/3: done.")

        logger.info("Stage 2/3: training the behavior-cloning policy.")
        policy_model = train_behavior_cloning_policy(
            config=config.policy,
            environment=environment,
            state_action_pairs=state_action_pairs,
            seed=config.seed,
            device=device,
            run=run,
        )
        logger.info("Stage 2/3: done.")

        logger.info("Stage 3/3: evaluating the policy.")
        evaluate_policy(
            environment=environment,
            policy_factory=create_behavior_cloning_policy_factory(
                policy_model=policy_model, device=device
            ),
            policy_label=f"bc ({run_name})",
            seed=config.seed,
            run=run,
        )
        logger.info("Stage 3/3: done.")


def _create_synthetic_state_action_pairs(
    *,
    augmentation_config: Augmentation,
    state_space: StateSpace,
    expert_episodes: EpisodeStore,
    seed: int,
    device: torch.device,
    run: Run,
) -> StateActionPairs:
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
                state_space=state_space,
                expert_episodes=expert_episodes,
                # Behavior cloning trains on single (state, action) pairs, not
                # windows, so no expert part is needed.
                expert_part_length=0,
                seed=seed,
                device=device,
                run=run,
            )
            return StateActionPairs(
                states=tacil_episodes.states, actions=tacil_episodes.actions
            )


if __name__ == "__main__":
    main()
