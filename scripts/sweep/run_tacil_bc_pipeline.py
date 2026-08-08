"""Pipeline for one sweep trial: TaCIL augmentation, behavior-cloning policy.

Same augmentation stages and swept parameters as run_pipeline (TaCIL + DP);
only the policy stage differs.
"""

import tempfile
from dataclasses import asdict
from pathlib import Path

import torch
import tyro
import wandb
from rich.pretty import pretty_repr

from scripts.data.augmentation.tacil.create_data import config as data_config
from scripts.data.augmentation.tacil.create_data.create_augmented_data import (
    create_augmented_data,
)
from scripts.data.augmentation.tacil.create_dynamics import config as dynamics_config
from scripts.data.augmentation.tacil.create_dynamics.train_dynamics_model_ensemble import (  # noqa: E501
    train_dynamics_model_ensemble,
)
from scripts.evaluation.config import DEFAULT_EPISODE_COUNT
from scripts.evaluation.core.condition import Condition
from scripts.evaluation.evaluate_policy import STAGE_PREFIX, evaluate_policy
from scripts.policies.bc import config as bc_config
from scripts.policies.bc.policy import load_behavior_cloning_policy
from scripts.policies.bc.train import train_behavior_cloning_policy
from scripts.shared.data import download_episode_store
from scripts.shared.log import create_logger, log_device_info
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH
from scripts.shared.seed import set_seed
from scripts.shared.wandb_config import WandbConfig
from scripts.sweep.config import Config
from scripts.sweep.run_pipeline import PASSED_IN_MEMORY

logger = create_logger()


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

    with wandb.init(
        project=config.wandb.project,
        notes=_build_note(config=config, condition=condition),
        job_type="tacil-bc-pipeline",
        config=asdict(obj=config),
        dir=OUTPUT_DIRECTORY_PATH,
    ) as run:
        expert_episode_store = download_episode_store(
            artifact_name=config.expert_dataset_artifact, run=run
        )

        logger.info("Stage 1/4: training the dynamics model ensemble.")
        dynamics_model_ensemble = train_dynamics_model_ensemble(
            config=_build_dynamics_config(config=config),
            episode_store=expert_episode_store,
            device=device,
            run=run,
        )
        logger.info("Stage 1/4: done.")

        logger.info("Stage 2/4: creating the augmented dataset.")
        augmented_episode_store = create_augmented_data(
            config=_build_data_config(config=config),
            episode_store=expert_episode_store,
            dynamics_model_ensemble=dynamics_model_ensemble,
            device=device,
            run=run,
        )
        logger.info("Stage 2/4: done.")

        with tempfile.TemporaryDirectory() as checkpoint_directory:
            logger.info("Stage 3/4: training the behavior-cloning policy.")
            checkpoint_path = train_behavior_cloning_policy(
                config=_build_policy_config(config=config),
                episode_store=augmented_episode_store,
                checkpoint_directory_path=Path(checkpoint_directory),
                device=device,
                run=run,
            )
            logger.info("Stage 3/4: done.")

            set_seed(seed=config.seed)
            logger.info("Stage 4/4: evaluating the policy.")
            evaluate_policy(
                policy_factory=load_behavior_cloning_policy(
                    checkpoint_path=checkpoint_path, device=device
                ),
                policy_label=f"bc   behavior-cloning-policy-{run.name}",
                condition=condition,
                episode_count=DEFAULT_EPISODE_COUNT,
                run=run,
            )
            logger.info("Stage 4/4: done.")

        reward_key = f"{STAGE_PREFIX}{condition.label}/reward_mean"
        logger.info("Trial finished: %s = %s", reward_key, run.summary[reward_key])


def _build_dynamics_config(*, config: Config) -> dynamics_config.Config:
    return dynamics_config.Config(
        wandb=WandbConfig(project=config.wandb.project),
        data=dynamics_config.DataConfig(
            artifact=config.expert_dataset_artifact,
            split=dynamics_config.DataSplitConfig(),
            loader=dynamics_config.DataLoaderConfig(window_length=config.window_length),
        ),
        architecture=dynamics_config.ArchitectureConfig(
            lipschitz_constant_upper_bound=config.lipschitz_constant_upper_bound,
            include_previous_state=config.include_previous_state,
            measure_positions_from_agent=config.measure_positions_from_agent,
        ),
        training=dynamics_config.TrainingConfig(),
        seed=config.seed,
    )


def _build_data_config(*, config: Config) -> data_config.Config:
    return data_config.Config(
        wandb=WandbConfig(project=config.wandb.project),
        model_artifact=PASSED_IN_MEMORY,
        dataset_artifact=config.expert_dataset_artifact,
        # Behavior cloning trains on single (state, action) pairs, so the
        # labels need no expert continuation behind them.
        expert_part_length=0,
        action_noise_fraction=config.action_noise_fraction,
        error_limit_fraction=config.error_limit_fraction,
        disagreement_weight=config.disagreement_weight,
        include_previous_state=config.include_previous_state,
        seed=config.seed,
    )


def _build_policy_config(*, config: Config) -> bc_config.Config:
    return bc_config.Config(
        wandb=WandbConfig(project=config.wandb.project),
        dimensionality=bc_config.DimensionalityConfig(),
        optimization=bc_config.OptimizationConfig(),
        evaluation=bc_config.EvaluationConfig(
            floor_condition=config.floor_condition,
            disturbance_level=config.disturbance_level,
        ),
        dataset_artifact=PASSED_IN_MEMORY,
        seed=config.seed,
    )


def _build_note(*, config: Config, condition: Condition) -> str:
    return (
        f"data={config.expert_dataset_artifact}"
        f"|lip={config.lipschitz_constant_upper_bound}"
        f"|win={config.window_length}"
        f"|seed={config.seed}"
        f"|noise={config.action_noise_fraction:.4f}"
        f"|err={config.error_limit_fraction:.4f}"
        f"|dw={config.disagreement_weight:.2f}"
        f"|prev={config.include_previous_state}"
        f"|rel={config.measure_positions_from_agent}"
        f"|cond={condition.label}"
    )


if __name__ == "__main__":
    main()
