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
from scripts.policies.dp import config as policy_config
from scripts.policies.dp.policy import load_diffusion_policy
from scripts.policies.dp.train import train_diffusion_policy
from scripts.shared.data import download_episode_store
from scripts.shared.log import create_logger, log_device_info
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH
from scripts.shared.seed import set_seed
from scripts.shared.wandb_config import WandbConfig
from scripts.sweep.config import Config

logger = create_logger()

# Stands in wherever a stage config names an input as an artifact that this
# process never publishes. Nothing reads it: the stages are handed their inputs.
PASSED_IN_MEMORY = "passed-in-memory"


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
        job_type="tacil-pipeline",
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

        policy_training_config = _build_policy_config(config=config)
        with tempfile.TemporaryDirectory() as checkpoint_directory:
            logger.info("Stage 3/4: training the diffusion policy.")
            checkpoint_path = train_diffusion_policy(
                config=policy_training_config,
                episode_store=augmented_episode_store,
                checkpoint_directory_path=Path(checkpoint_directory),
                device=device,
                run=run,
            )
            logger.info("Stage 3/4: done.")

            # Score the selected checkpoint the way it was selected.
            sampler = policy_training_config.evaluation.sampler
            set_seed(seed=config.seed)
            logger.info("Stage 4/4: evaluating the policy.")
            evaluate_policy(
                policy_factory=load_diffusion_policy(
                    checkpoint_path=checkpoint_path, sampler=sampler, device=device
                ),
                policy_label=f"dp ({sampler})   diffusion-policy-{run.name}",
                condition=condition,
                episode_count=DEFAULT_EPISODE_COUNT,
                run=run,
            )
            logger.info("Stage 4/4: done.")

        # What the sweep maximizes.
        reward_key = f"{STAGE_PREFIX}{condition.label}/reward_mean"
        logger.info("Trial finished: %s = %s", reward_key, run.summary[reward_key])


# The stage configs carry the wandb settings and artifact names their own
# command lines need. The stage functions ignore both: they are handed their
# inputs and the run to log to.


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
        action_noise_fraction=config.action_noise_fraction,
        error_limit_fraction=config.error_limit_fraction,
        disagreement_weight=config.disagreement_weight,
        include_previous_state=config.include_previous_state,
        seed=config.seed,
    )


def _build_policy_config(*, config: Config) -> policy_config.Config:
    return policy_config.Config(
        wandb=WandbConfig(project=config.wandb.project),
        dataset_artifact=PASSED_IN_MEMORY,
        dimensionality=policy_config.DimensionalityConfig(),
        horizon_length=policy_config.HorizonLengthConfig(),
        optimization=policy_config.OptimizationConfig(),
        evaluation=policy_config.EvaluationConfig(
            floor_condition=config.floor_condition,
            disturbance_level=config.disturbance_level,
        ),
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
