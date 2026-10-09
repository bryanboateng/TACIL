import dataclasses
from dataclasses import asdict, dataclass

import torch
import tyro
import wandb
from rich.pretty import pretty_repr

from scripts.policies.mopo import config as mopo_config
from scripts.policies.mopo.replication.dataset import load_dataset
from scripts.policies.mopo.replication.evaluation import evaluate
from scripts.policies.mopo.replication.results import write_result
from scripts.policies.mopo.replication.tasks import ALL_TASKS, EPOCH_COUNT, Task
from scripts.policies.mopo.replication.termination import get_termination_function
from scripts.policies.mopo.train import train_mopo_policy
from scripts.policies.mopo.transitions import Transitions
from scripts.shared.log import create_logger, log_device_info
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH

logger = create_logger()


@dataclass(frozen=True)
class Config:
    seed: int
    wandb_project: str

    task: str

    final_episode_count: int = 100

    real_data_ratio: float | None = None
    epoch_count: int | None = None

    def __post_init__(self) -> None:
        if self.task not in ALL_TASKS:
            raise ValueError(
                f"task must be one of {sorted(ALL_TASKS)}, but it is {self.task!r}."
            )


def main() -> None:
    config = tyro.cli(f=Config)
    logger.info(pretty_repr(config))
    _run(config=config)


def _run(*, config: Config) -> None:
    task = ALL_TASKS[config.task]

    device = torch.device(device="cuda" if torch.cuda.is_available() else "cpu")
    log_device_info(device=device, logger=logger)

    policy_config = _build_policy_config(task=task, config=config)
    logger.info(pretty_repr(policy_config))

    with wandb.init(
        project=config.wandb_project,
        job_type="mopo-replication",
        config=asdict(obj=config) | {"policy": asdict(obj=policy_config)},
        dir=OUTPUT_DIRECTORY_PATH,
    ) as run:
        run.summary["0_task/1_name"] = task.name
        run.summary["0_task/2_published_score"] = task.published_score
        run.summary["0_task/3_published_score_std"] = task.published_score_std
        run.summary["0_task/4_rollout_length"] = task.rollout_length
        run.summary["0_task/5_penalty_coefficient"] = task.penalty_coefficient

        logger.info("Stage 1/3: reading %s.", task.dataset_file_name)
        dataset = load_dataset(task=task)
        transitions = Transitions(
            states=torch.from_numpy(dataset.states).to(device=device),
            actions=torch.from_numpy(dataset.actions).to(device=device),
            next_states=torch.from_numpy(dataset.next_states).to(device=device),
            rewards=torch.from_numpy(dataset.rewards).to(device=device),
            terminations=torch.from_numpy(dataset.terminations).to(device=device),
        )
        run.summary["0_dataset/1_transition_count"] = len(transitions.states)
        run.summary["0_dataset/2_state_dimensionality"] = dataset.state_dimensionality
        run.summary["0_dataset/3_action_dimensionality"] = dataset.action_dimensionality
        run.summary["0_dataset/4_reward_mean"] = float(transitions.rewards.mean())
        run.summary["0_dataset/5_termination_count"] = int(
            transitions.terminations.sum()
        )
        logger.info(
            "Stage 1/3: %d transitions, state %d, action %d.",
            len(transitions.states),
            dataset.state_dimensionality,
            dataset.action_dimensionality,
        )

        logger.info("Stage 2/3: training the MOPO policy.")
        policy = train_mopo_policy(
            config=policy_config,
            transitions=transitions,
            termination_function=get_termination_function(robot=task.robot),
            validation=None,
            seed=config.seed,
            device=device,
            run=run,
        )
        logger.info("Stage 2/3: done.")

        logger.info("Stage 3/3: evaluating in %s.", task.environment_id)
        result = evaluate(
            policy=policy,
            task=task,
            episode_count=config.final_episode_count,
            seed=config.seed + 500_000,
            device=device,
        )
        run.summary["4_evaluation/1_score_mean"] = result.score_mean
        run.summary["4_evaluation/2_score_std"] = result.score_std
        run.summary["4_evaluation/3_return_mean"] = result.return_mean
        run.summary["4_evaluation/4_episode_length_mean"] = result.episode_length_mean
        if task.published_score is not None and task.published_score_std is not None:
            run.summary["4_evaluation/5_published_score"] = task.published_score
            run.summary["4_evaluation/6_score_difference"] = (
                result.score_mean - task.published_score
            )
            run.summary["4_evaluation/7_within_published_spread"] = (
                abs(result.score_mean - task.published_score)
                <= task.published_score_std
            )
        result_file_path = write_result(
            task=task.name, seed=config.seed, score=result.score_mean
        )
        logger.info(
            "Stage 3/3: score %.1f +- %.1f, published %s. Wrote %s.",
            result.score_mean,
            result.score_std,
            task.published_score,
            result_file_path,
        )


# ---- Private --------------------------------------------


def _build_policy_config(*, task: Task, config: Config) -> mopo_config.Config:
    default = mopo_config.Config(
        dynamics=mopo_config.DynamicsConfig(),
        rollout=mopo_config.RolloutConfig(),
        optimization=mopo_config.OptimizationConfig(),
        validation=None,
    )
    return dataclasses.replace(
        default,
        rollout=dataclasses.replace(
            default.rollout,
            length=task.rollout_length,
            penalty_coefficient=task.penalty_coefficient,
        ),
        optimization=dataclasses.replace(
            default.optimization,
            epoch_count=config.epoch_count or EPOCH_COUNT,
            real_data_ratio=(
                default.optimization.real_data_ratio
                if config.real_data_ratio is None
                else config.real_data_ratio
            ),
        ),
        validation=None,
    )


if __name__ == "__main__":
    main()
