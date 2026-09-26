from typing import TypeAlias

import numpy as np
import torch
import tyro
import wandb
from rich.pretty import pretty_repr

from scripts.environments.environment import Environment
from scripts.environments.registry import create_environment
from scripts.evaluation.config import VALIDATION_START_SEED
from scripts.evaluation.core.scoring import (
    calculate_reward_statistics,
    calculate_success_statistics,
    roll_out_episodes_with_separate_seeds_per_condition,
)
from scripts.evaluation.evaluate_policy import evaluate_policy
from scripts.policies.mopo import config as mopo_config
from scripts.policies.mopo.policy import MopoPolicy, create_mopo_policy_factory
from scripts.policies.mopo.train import ValidationScore, train_mopo_policy
from scripts.policies.mopo.transitions import Transitions
from scripts.policies.shared.run_config import RunConfig, build_run_record
from scripts.shared.dataset_artifact import download_episodes
from scripts.shared.episodes import RewardedEpisodes, calculate_indices_with_successor
from scripts.shared.log import create_logger, log_device_info
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
        run_name = run.name
        if run_name is None:
            raise RuntimeError("wandb run name unexpectedly None")

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

        policy = train_mopo_policy(
            config=config.policy,
            transitions=transitions,
            termination_function=lambda state_batch: _calculate_termination(
                environment=environment,
                normalized_state_batch=state_batch,
                state_normalizer=state_normalizer,
            ),
            score_policy=lambda policy: _score_policy(
                environment=environment,
                policy=policy,
                state_normalizer=state_normalizer,
                action_normalizer=action_normalizer,
                episode_count_per_condition=config.policy.validation.episode_count_per_condition,
                parallel_environment_count=config.policy.validation.parallel_environment_count,
                device=device,
            ),
            selection_metric=environment.checkpoint_selection_metric,
            seed=config.seed,
            device=device,
            run=run,
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
            policy_label=f"mopo ({run_name})",
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


def _calculate_termination(
    *,
    environment: Environment,
    normalized_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    state_normalizer: ZScoreNormalizer,
) -> torch.Tensor:  # (batch_size,)
    state_batch = (
        state_normalizer.denormalize(value_batch=normalized_state_batch.cpu())
        .numpy()
        .astype(np.float32)
    )  # (batch_size, state_dimensionality)

    return torch.from_numpy(
        environment.calculate_termination(state_batch=state_batch)
    ).to(device=normalized_state_batch.device)


def _score_policy(
    *,
    environment: Environment,
    policy: MopoPolicy,
    state_normalizer: ZScoreNormalizer,
    action_normalizer: MinMaxNormalizer,
    episode_count_per_condition: int,
    parallel_environment_count: int,
    device: torch.device,
) -> ValidationScore:
    episode_records = roll_out_episodes_with_separate_seeds_per_condition(
        environment=environment,
        policy_factory=create_mopo_policy_factory(
            policy=policy,
            state_normalizer=state_normalizer,
            action_normalizer=action_normalizer,
            device=device,
        ),
        start_episode_seed=VALIDATION_START_SEED,
        episode_count_per_condition=episode_count_per_condition,
        parallel_environment_count=parallel_environment_count,
    )
    reward_statistics = calculate_reward_statistics(episode_records=episode_records)
    return ValidationScore(
        mean=reward_statistics.mean,
        std=reward_statistics.std,
        success_rate=calculate_success_statistics(episode_records=episode_records).rate,
    )


if __name__ == "__main__":
    main()
