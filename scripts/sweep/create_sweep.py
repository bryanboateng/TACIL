"""Create a pipeline sweep.

The sweep searches the whole chain with Bayesian optimization, judged by the
reward in one condition. Every sweep is the same apart from the dataset it
starts from and the condition it optimizes, which is what this script takes.

    uv run python -m scripts.sweep.create_sweep \
        --project-prefix=2026-08-01 --dataset=full \
        --floor-condition=GRIPPY --disturbance-level=None

It prints the command that puts an agent on the sweep. Run that as often as
there are GPUs to spare; agents can join and leave while the sweep runs.
"""

from dataclasses import dataclass
from typing import Any, Literal

import tyro
import wandb
from rich.pretty import pretty_repr

from scripts.evaluation.core.condition import (
    Condition,
    DisturbanceLevel,
    FloorCondition,
)
from scripts.shared.log import create_logger

logger = create_logger()

DatasetLabel = Literal["full", "sub025"]

# The datasets a sweep can start from, under the label that names the project.
EXPERT_DATASET_ARTIFACTS: dict[DatasetLabel, str] = {
    "full": "bryanboateng-team/datasets/expert-dataset:v0",
    "sub025": "bryanboateng-team/datasets/subsampled-dataset-magic-wind-2:v0",
}


@dataclass(frozen=True)
class Config:
    # Names the project, together with the dataset and the condition.
    project_prefix: str
    dataset: DatasetLabel
    floor_condition: FloorCondition
    disturbance_level: DisturbanceLevel | None

    entity: str = "bryanboateng-team"


def main() -> None:
    config = tyro.cli(f=Config, config=[tyro.conf.UsePythonSyntaxForLiteralCollections])
    logger.info(pretty_repr(config))

    condition = Condition(
        floor_condition=config.floor_condition,
        disturbance_level=config.disturbance_level,
    )
    project = f"{config.project_prefix}-{config.dataset}-{condition.label}"

    sweep_configuration = _build_sweep_configuration(
        config=config, condition=condition, project=project
    )
    logger.info(pretty_repr(sweep_configuration))

    sweep_id = wandb.sweep(
        sweep=sweep_configuration, entity=config.entity, project=project
    )
    sweep_path = f"{config.entity}/{project}/{sweep_id}"

    # Keep the -f: without it the agent runs the trial in a process group of its
    # own, where a Ctrl-C typed in the terminal never reaches it.
    logger.info(
        "Add an agent:  CUDA_VISIBLE_DEVICES=0 uv run wandb agent -f %s", sweep_path
    )
    logger.info(
        "Stop the sweep and its agents:  uv run wandb sweep --cancel %s", sweep_path
    )


def _build_sweep_configuration(
    *, config: Config, condition: Condition, project: str
) -> dict[str, Any]:
    return {
        "program": "scripts/sweep/run_pipeline.py",
        "method": "bayes",
        "metric": {"name": f"{condition.label}/reward_mean", "goal": "maximize"},
        # run_pipeline is a module, not a script: it imports from the repository
        # root, which is only on the path when python is started with -m.
        "command": [
            "${env}",
            "${interpreter}",
            "-m",
            "scripts.sweep.run_pipeline",
            f"--wandb.project={project}",
            f"--expert-dataset-artifact={EXPERT_DATASET_ARTIFACTS[config.dataset]}",
            f"--floor-condition={config.floor_condition.name}",
            f"--disturbance-level={_disturbance_level_argument(config=config)}",
            "${args}",
        ],
        "parameters": {
            # Shape the dynamics model. "none" leaves it unconstrained.
            "lipschitz_constant_upper_bound": {
                "values": ["none", "1", "2", "5", "10", "20", "50", "100"]
            },
            "window_length": {"distribution": "int_uniform", "min": 1, "max": 15},
            # Shape the backward rollout.
            "action_noise_fraction": {
                "distribution": "uniform",
                "min": 0.0,
                "max": 1.0,
            },
            # A multiple of the median expert step, so it spans orders of
            # magnitude.
            "error_limit_fraction": {
                "distribution": "log_uniform_values",
                "min": 0.125,
                "max": 32.0,
            },
        },
    }


def _disturbance_level_argument(*, config: Config) -> str:
    level = config.disturbance_level
    return "None" if level is None else level.name


if __name__ == "__main__":
    main()
