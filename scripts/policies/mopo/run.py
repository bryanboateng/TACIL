from dataclasses import asdict
from typing import TypeAlias

import numpy as np
import torch
import tyro
import wandb
from rich.pretty import pretty_repr

from scripts.environments.environment import Environment
from scripts.environments.registry import create_environment
from scripts.evaluation.core.scoring import (
    calculate_reward_statistics,
    run_scored_episodes_over_conditions,
)
from scripts.evaluation.evaluate_policy import evaluate_policy
from scripts.policies.mopo import config as mopo_config
from scripts.policies.mopo.policy import MopoPolicy, create_mopo_policy_factory
from scripts.policies.mopo.train import ValidationScore, train_mopo_policy
from scripts.policies.mopo.transitions import Transitions
from scripts.policies.shared.run_config import RunConfig
from scripts.shared.dataset_artifact import download_episode_store
from scripts.shared.episode_store import EpisodeStore
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
        config=asdict(obj=config),
        dir=OUTPUT_DIRECTORY_PATH,
    ) as run:
        run_name = run.name
        if run_name is None:
            raise RuntimeError("wandb run name unexpectedly None")

        environment = create_environment(name=config.environment)

        expert_episodes = download_episode_store(
            artifact_name=config.expert_dataset_artifact, run=run
        )

        logger.info("Stage 1/3: building the training data.")
        state_normalizer = ZScoreNormalizer(
            value_batch=torch.from_numpy(expert_episodes.states).float()
        )
        action_normalizer = MinMaxNormalizer(
            value_batch=torch.from_numpy(expert_episodes.actions).float()
        )
        transitions = _build_transitions(
            environment=environment,
            episode_store=expert_episodes,
            state_normalizer=state_normalizer,
            action_normalizer=action_normalizer,
            device=device,
        )
        run.summary["0_transitions/1_count"] = len(transitions.states)
        run.summary["0_transitions/2_reward_mean"] = float(transitions.rewards.mean())
        run.summary["0_transitions/3_reward_max"] = float(transitions.rewards.max())
        logger.info("Stage 1/3: done.")

        logger.info("Stage 2/3: training the MOPO policy.")
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
                device=device,
            ),
            seed=config.seed,
            device=device,
            run=run,
        )
        logger.info("Stage 2/3: done.")

        logger.info("Stage 3/3: evaluating the policy.")
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
            record_videos=config.record_videos,
            run=run,
        )
        logger.info("Stage 3/3: done.")


# ---- Private --------------------------------------------


def _build_transitions(
    *,
    environment: Environment,
    episode_store: EpisodeStore,
    state_normalizer: ZScoreNormalizer,
    action_normalizer: MinMaxNormalizer,
    device: torch.device,
) -> Transitions:
    has_successor = np.ones(shape=len(episode_store.states), dtype=np.bool_)
    has_successor[episode_store.episode_end_indices - 1] = False
    source_indices = np.flatnonzero(has_successor)

    next_states = episode_store.states[source_indices + 1]

    return Transitions(
        states=state_normalizer.normalize(
            value_batch=torch.from_numpy(episode_store.states[source_indices]).float()
        ).to(device=device),
        actions=action_normalizer.normalize(
            value_batch=torch.from_numpy(episode_store.actions[source_indices]).float()
        ).to(device=device),
        next_states=state_normalizer.normalize(
            value_batch=torch.from_numpy(next_states).float()
        ).to(device=device),
        rewards=torch.from_numpy(
            environment.calculate_transition_rewards(
                episode_store=episode_store, transition_indices=source_indices
            ).reshape(-1, 1)
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
    device: torch.device,
) -> ValidationScore:
    reward_statistics = calculate_reward_statistics(
        episode_records=run_scored_episodes_over_conditions(
            environment=environment,
            policy_factory=create_mopo_policy_factory(
                policy=policy,
                state_normalizer=state_normalizer,
                action_normalizer=action_normalizer,
                device=device,
            ),
            episode_count_per_condition=episode_count_per_condition,
        )
    )
    return ValidationScore(mean=reward_statistics.mean, std=reward_statistics.std)


if __name__ == "__main__":
    main()
