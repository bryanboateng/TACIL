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
from scripts.evaluation.evaluate_policy import STAGE_NAME
from scripts.shared.log import create_logger

logger = create_logger()

DatasetLabel = Literal["full", "sub025"]

SearchLabel = Literal["bayes", "random"]

MethodLabel = Literal["tacil-dp", "tacil-bc", "ccil-bc-noisy", "ccil-bc-euler", "gpi"]

METHOD_MODULES: dict[MethodLabel, str] = {
    "tacil-dp": "scripts.policies.dp.run",
    "tacil-bc": "scripts.policies.bc.run",
    "ccil-bc-noisy": "scripts.policies.bc.run",
    "ccil-bc-euler": "scripts.policies.bc.run",
    "gpi": "scripts.policies.gpi.run",
}

METHOD_FIXED_ARGUMENTS: dict[MethodLabel, list[str]] = {
    "tacil-dp": ["augmentation:tacil"],
    "tacil-bc": ["augmentation:tacil"],
    "ccil-bc-noisy": ["augmentation:ccil", "--augmentation.labels.type=noisy_action"],
    "ccil-bc-euler": [
        "augmentation:ccil",
        "--augmentation.labels.type=backward_euler_fast",
    ],
}

_TACIL_PARAMETERS: dict[str, Any] = {
    "augmentation.dynamics.lipschitz_constant_upper_bound": {
        "values": [None, 1, 2, 5, 10, 20, 50, 100]
    },
    "augmentation.dynamics.window_length": {
        "distribution": "int_uniform",
        "min": 1,
        "max": 15,
    },
    "augmentation.data.action_noise_fraction": {
        "distribution": "uniform",
        "min": 0.0,
        "max": 1.0,
    },
    # The limit is a fraction of the median expert step. The replay error is a
    # real distance on the same scale, so the range spans "a hair off course"
    # to "several steps off course" — the old signal-dependent extremes are
    # not needed anymore.
    "augmentation.data.error_limit_fraction": {
        "distribution": "log_uniform_values",
        "min": 0.01,
        "max": 10.0,
    },
}

_CCIL_SHARED_PARAMETERS: dict[str, Any] = {
    "augmentation.dynamics.lipschitz_type": {
        "values": ["soft_sampling", "spectral_normalization", "none"]
    },
    "augmentation.dynamics.lipschitz_constraint": {
        "values": [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    },
    "augmentation.labels.rejection_quantile": {
        "distribution": "uniform",
        "min": 0.0,
        "max": 1.0,
    },
}

METHOD_PARAMETERS: dict[MethodLabel, dict[str, Any]] = {
    "tacil-dp": _TACIL_PARAMETERS,
    "tacil-bc": _TACIL_PARAMETERS,
    "ccil-bc-noisy": {
        **_CCIL_SHARED_PARAMETERS,
        "augmentation.labels.action_noise_std": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 10.0,
        },
    },
    "ccil-bc-euler": _CCIL_SHARED_PARAMETERS,
    "gpi": {
        "policy.progression_weight": {
            "distribution": "log_uniform_values",
            "min": 0.1,
            "max": 10.0,
        },
        "policy.attraction_weight": {
            "distribution": "log_uniform_values",
            "min": 0.1,
            "max": 10.0,
        },
        "policy.plan_length": {"distribution": "int_uniform", "min": 1, "max": 16},
    },
}

EXPERT_DATASET_ARTIFACTS: dict[DatasetLabel, str] = {
    "full": "bryanboateng-team/datasets/expert-dataset:v0",
    "sub025": "bryanboateng-team/datasets/subsampled-dataset-0.25:v0",
}


@dataclass(frozen=True)
class Config:
    project_prefix: str
    dataset: DatasetLabel
    method: MethodLabel
    floor_condition: FloorCondition
    disturbance_level: DisturbanceLevel | None

    seed: int

    sweep_method: SearchLabel = "bayes"

    entity: str = "bryanboateng-team"


def main() -> None:
    config = tyro.cli(f=Config, config=[tyro.conf.UsePythonSyntaxForLiteralCollections])
    logger.info(pretty_repr(config))

    condition = Condition(
        floor_condition=config.floor_condition,
        disturbance_level=config.disturbance_level,
    )
    project = (
        f"{config.project_prefix}-{config.dataset}-{config.method}-{condition.label}"
    )

    sweep_configuration = _build_sweep_configuration(
        config=config, condition=condition, project=project
    )
    logger.info(pretty_repr(sweep_configuration))

    sweep_id = wandb.sweep(
        sweep=sweep_configuration, entity=config.entity, project=project
    )

    logger.info("Sweep path: %s", f"{config.entity}/{project}/{sweep_id}")


def _build_sweep_configuration(
    *, config: Config, condition: Condition, project: str
) -> dict[str, Any]:
    pipeline_module = METHOD_MODULES[config.method]
    return {
        "program": pipeline_module.replace(".", "/") + ".py",
        "method": config.sweep_method,
        "metric": {
            "name": f"{STAGE_NAME}/{condition.label}/reward_mean",
            "goal": "maximize",
        },
        "command": [
            "${env}",
            "${interpreter}",
            "-m",
            pipeline_module,
            f"--wandb-project={project}",
            f"--expert-dataset-artifact={EXPERT_DATASET_ARTIFACTS[config.dataset]}",
            f"--floor-condition={config.floor_condition.name}",
            f"--disturbance-level={'None' if config.disturbance_level is None else config.disturbance_level.name}",
            f"--seed={config.seed}",
            *METHOD_FIXED_ARGUMENTS.get(config.method, []),
            "${args_no_boolean_flags}",
        ],
        "parameters": METHOD_PARAMETERS[config.method],
    }


if __name__ == "__main__":
    main()
