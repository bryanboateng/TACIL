from typing import TypeAlias

import numpy as np
import torch
import tyro
import wandb
from rich.pretty import pretty_repr

from scripts.data.augmentation.augment import (
    AugmentationConfig,
    augment,
    get_augmentation_name,
)
from scripts.environments.registry import create_environment
from scripts.evaluation.evaluate_policy import evaluate_policy
from scripts.policies.bc import config as bc_config
from scripts.policies.bc.policy import create_behavior_cloning_policy_factory
from scripts.policies.bc.train import train_behavior_cloning_policy
from scripts.policies.shared.run_config import RunConfig, build_run_record
from scripts.policies.shared.validation import Validation
from scripts.shared.dataset_artifact import download_episodes
from scripts.shared.episodes import drop_rewards
from scripts.shared.log import create_logger, log_device_info
from scripts.shared.model_artifact import POLICY_ARTIFACT_TYPE, upload_model
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH
from scripts.shared.state_action_transitions import StateActionTransitions

logger = create_logger()

Config: TypeAlias = RunConfig[bc_config.Config, AugmentationConfig]


def main() -> None:
    config = tyro.cli(f=Config)
    logger.info(pretty_repr(config))
    _run(config=config)


def _run(*, config: Config) -> None:
    device = torch.device(device="cuda" if torch.cuda.is_available() else "cpu")
    log_device_info(device=device, logger=logger)

    with wandb.init(
        project=config.wandb_project,
        job_type="bc",
        config=build_run_record(
            config=config,
            policy_name="bc",
            augmentation_name=get_augmentation_name(config=config.augmentation),
        ),
        dir=OUTPUT_DIRECTORY_PATH,
    ) as run:
        environment = create_environment(name=config.environment)

        dataset = download_episodes(
            artifact_name=config.expert_dataset_artifact, run=run
        )
        run.config["expert_dataset_artifact_version"] = dataset.artifact_version
        expert_episodes = drop_rewards(episodes=dataset.episodes)

        logger.info("Stage 1/3: building the training data.")
        expert_state_action_pairs = StateActionTransitions(
            states=expert_episodes.states, actions=expert_episodes.actions
        )
        if config.augmentation is None:
            state_action_transitions = expert_state_action_pairs
        else:
            augmented_episodes = augment(
                config=config.augmentation,
                environment=environment,
                expert_episodes=expert_episodes,
                # Behavior cloning trains on single (state, action) pairs, not
                # windows, so no expert part is needed.
                expert_part_length=0,
                seed=config.seed,
                device=device,
                run=run,
            )
            state_action_transitions = StateActionTransitions(
                states=np.concatenate(
                    [expert_state_action_pairs.states, augmented_episodes.states],
                    axis=0,
                ),  # (pair_count, state_dimensionality)
                actions=np.concatenate(
                    [expert_state_action_pairs.actions, augmented_episodes.actions],
                    axis=0,
                ),  # (pair_count, action_dimensionality)
            )

        logger.info("Stage 1/3: done.")

        logger.info("Stage 2/3: training the behavior-cloning policy.")
        policy_model = train_behavior_cloning_policy(
            config=config.policy,
            validation=(
                None
                if config.policy.validation is None
                else Validation(
                    settings=config.policy.validation, environment=environment
                )
            ),
            state_action_transitions=state_action_transitions,
            seed=config.seed,
            device=device,
            run=run,
        )
        if config.policy.save_model:
            upload_model(
                model=policy_model, artifact_type=POLICY_ARTIFACT_TYPE, run=run
            )
        logger.info("Stage 2/3: done.")

        logger.info("Stage 3/3: evaluating the policy.")
        evaluate_policy(
            environment=environment,
            policy_factory=create_behavior_cloning_policy_factory(
                policy_model=policy_model, device=device
            ),
            config=config.evaluation,
            run=run,
        )
        logger.info("Stage 3/3: done.")


if __name__ == "__main__":
    main()
