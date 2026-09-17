from dataclasses import asdict
from typing import TypeAlias

import numpy as np
import torch
import tyro
import wandb
from rich.pretty import pretty_repr

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
from scripts.shared.pusht_reward import calculate_reward, calculate_termination
from scripts.shared.pusht_state import ARENA_SIZE, POSITION_SLICE

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
                normalized_state_batch=state_batch, state_normalizer=state_normalizer
            ),
            score_policy=lambda policy: _score_policy(
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
    episode_store: EpisodeStore,
    state_normalizer: ZScoreNormalizer,
    action_normalizer: MinMaxNormalizer,
    device: torch.device,
) -> Transitions:
    has_successor = np.ones(shape=len(episode_store.states), dtype=np.bool_)
    has_successor[episode_store.episode_end_indices - 1] = False
    source_indices = np.flatnonzero(has_successor)

    next_states = episode_store.states[source_indices + 1]

    # The rewards come from the board geometry, which is worked out on the
    # processor. Everything is moved over once, here.
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
            calculate_reward(state_batch=next_states).astype(np.float32).reshape(-1, 1)
        ).to(device=device),
        terminations=torch.from_numpy(
            calculate_termination(state_batch=next_states)
            .astype(np.float32)
            .reshape(-1, 1)
        ).to(device=device),
    )


def _calculate_termination(
    *,
    normalized_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    state_normalizer: ZScoreNormalizer,
) -> torch.Tensor:  # (batch_size,)
    state_batch = (
        state_normalizer.denormalize(value_batch=normalized_state_batch.cpu())
        .numpy()
        .astype(np.float32)
    )

    position_batch = state_batch[:, POSITION_SLICE]
    plausible_batch = (
        np.isfinite(state_batch).all(axis=-1)
        & (position_batch > -ARENA_SIZE).all(axis=-1)
        & (position_batch < 2 * ARENA_SIZE).all(axis=-1)
    )

    termination_batch = ~plausible_batch
    if plausible_batch.any():
        termination_batch[plausible_batch] = calculate_termination(
            state_batch=state_batch[plausible_batch]
        )
    return torch.from_numpy(termination_batch).to(device=normalized_state_batch.device)


def _score_policy(
    *,
    policy: MopoPolicy,
    state_normalizer: ZScoreNormalizer,
    action_normalizer: MinMaxNormalizer,
    episode_count_per_condition: int,
    device: torch.device,
) -> ValidationScore:
    reward_statistics = calculate_reward_statistics(
        episode_records=run_scored_episodes_over_conditions(
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
