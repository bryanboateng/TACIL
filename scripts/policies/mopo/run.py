from dataclasses import asdict
from typing import TypeAlias

import torch
import tyro
import wandb
from rich.pretty import pretty_repr

from scripts.evaluation.evaluate_policy import evaluate_policy
from scripts.policies.mopo import config as mopo_config
from scripts.policies.mopo.policy import create_mopo_policy_factory
from scripts.policies.mopo.train import train_mopo_policy
from scripts.policies.mopo.transitions import build_transitions
from scripts.policies.shared.run_config import RunConfig
from scripts.shared.episode_store import download_episode_store
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
        transitions = build_transitions(
            episode_store=expert_episodes,
            state_normalizer=state_normalizer,
            action_normalizer=action_normalizer,
        )
        run.summary["0_transitions/1_count"] = len(transitions.states)
        run.summary["0_transitions/2_reward_mean"] = float(transitions.rewards.mean())
        run.summary["0_transitions/3_reward_max"] = float(transitions.rewards.max())
        logger.info("Stage 1/3: done.")

        logger.info("Stage 2/3: training the MOPO policy.")
        policy = train_mopo_policy(
            config=config.policy,
            transitions=transitions,
            state_normalizer=state_normalizer,
            action_normalizer=action_normalizer,
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
            seed=config.seed,
            run=run,
        )
        logger.info("Stage 3/3: done.")


if __name__ == "__main__":
    main()
