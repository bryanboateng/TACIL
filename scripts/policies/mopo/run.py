from typing import TypeAlias

import numpy as np
import torch
import tyro
import wandb
from rich.pretty import pretty_repr

from scripts.environments.environment import Environment
from scripts.environments.registry import create_environment
from scripts.evaluation.evaluate_policy import evaluate_policy
from scripts.policies.mopo import config as mopo_config
from scripts.policies.mopo.policy import create_mopo_policy_factory
from scripts.policies.mopo.train import (
    NormalizedEnvironment,
    create_termination_function,
    train_mopo_policy,
)
from scripts.policies.mopo.transitions import Transitions
from scripts.policies.shared.run_config import RunConfig, build_run_record
from scripts.policies.shared.validation import Validation
from scripts.shared.dataset_artifact import download_episodes
from scripts.shared.episodes import RewardedEpisodes, calculate_indices_with_successor
from scripts.shared.log import create_logger, log_device_info
from scripts.shared.model_artifact import POLICY_ARTIFACT_TYPE, upload_model
from scripts.shared.normalization import MinMaxNormalizer, ZScoreNormalizer
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH

logger = create_logger()

Config: TypeAlias = RunConfig[mopo_config.Config, None]


def main() -> None:
    config = tyro.cli(f=Config)
    logger.info(pretty_repr(config))
    _run(config=config)


def _run(*, config: Config) -> None:
    device = torch.device(device="cuda" if torch.cuda.is_available() else "cpu")
    log_device_info(device=device, logger=logger)

    with wandb.init(
        project=config.wandb_project,
        job_type="mopo",
        config=build_run_record(
            config=config, policy_name="mopo", augmentation_name=None
        ),
        dir=OUTPUT_DIRECTORY_PATH,
    ) as run:
        environment = create_environment(name=config.environment)

        dataset = download_episodes(
            artifact_name=config.expert_dataset_artifact, run=run
        )
        run.config["expert_dataset_artifact_version"] = dataset.artifact_version
        expert_episodes = dataset.episodes

        state_normalizer = ZScoreNormalizer(
            value_batch=torch.from_numpy(expert_episodes.states).float()
        )
        action_normalizer = MinMaxNormalizer(
            value_batch=torch.from_numpy(expert_episodes.actions).float()
        )
        transitions = _build_transitions(
            environment=environment,
            episodes=expert_episodes,
            state_normalizer=state_normalizer,
            action_normalizer=action_normalizer,
            device=device,
        )

        logger.info("Stage 1/2: training the MOPO policy.")

        normalized_environment = NormalizedEnvironment(
            environment=environment,
            state_normalizer=state_normalizer,
            action_normalizer=action_normalizer,
        )
        policy = train_mopo_policy(
            config=config.policy,
            transitions=transitions,
            termination_function=create_termination_function(
                normalized_environment=normalized_environment
            ),
            validation=(
                None
                if config.policy.validation is None
                else Validation(
                    settings=config.policy.validation,
                    environment=normalized_environment,
                )
            ),
            seed=config.seed,
            device=device,
            run=run,
        )
        if config.policy.save_model:
            upload_model(
                model=policy.actor, artifact_type=POLICY_ARTIFACT_TYPE, run=run
            )
        logger.info("Stage 1/2: done.")

        logger.info("Stage 2/2: evaluating the policy.")
        evaluate_policy(
            environment=environment,
            policy_factory=create_mopo_policy_factory(
                policy=policy,
                state_normalizer=state_normalizer,
                action_normalizer=action_normalizer,
                device=device,
            ),
            config=config.evaluation,
            run=run,
        )
        logger.info("Stage 2/2: done.")


# ---- Private --------------------------------------------


def _build_transitions(
    *,
    environment: Environment,
    episodes: RewardedEpisodes,
    state_normalizer: ZScoreNormalizer,
    action_normalizer: MinMaxNormalizer,
    device: torch.device,
) -> Transitions:
    indices_with_successor = calculate_indices_with_successor(
        episode_end_indices=episodes.episode_end_indices,
        total_timestep_count=len(episodes.states),
    )  # (transition_count,)

    next_states = episodes.states[indices_with_successor + 1]

    return Transitions(
        states=state_normalizer.normalize(
            value_batch=torch.from_numpy(
                episodes.states[indices_with_successor]
            ).float()
        ).to(device=device),
        actions=action_normalizer.normalize(
            value_batch=torch.from_numpy(
                episodes.actions[indices_with_successor]
            ).float()
        ).to(device=device),
        next_states=state_normalizer.normalize(
            value_batch=torch.from_numpy(next_states).float()
        ).to(device=device),
        rewards=torch.from_numpy(
            episodes.rewards[indices_with_successor].reshape(-1, 1)
        ).to(device=device),
        terminations=torch.from_numpy(
            environment.calculate_termination(state_batch=next_states)
            .astype(np.float32)
            .reshape(-1, 1)
        ).to(device=device),
    )


if __name__ == "__main__":
    main()
