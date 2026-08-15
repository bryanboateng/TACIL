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

SearchLabel = Literal["bayes", "random"]

MethodLabel = Literal[
    "tacil-dp-bounded",
    "tacil-dp-unbounded",
    "tacil-bc-bounded",
    "tacil-bc-unbounded",
    "ccil-bc-noisy",
    "ccil-bc-euler",
    "gpi",
]

METHOD_MODULES: dict[MethodLabel, str] = {
    "tacil-dp-bounded": "scripts.policies.dp.run",
    "tacil-dp-unbounded": "scripts.policies.dp.run",
    "tacil-bc-bounded": "scripts.policies.bc.run",
    "tacil-bc-unbounded": "scripts.policies.bc.run",
    "ccil-bc-noisy": "scripts.policies.bc.run",
    "ccil-bc-euler": "scripts.policies.bc.run",
    "gpi": "scripts.policies.gpi.run",
}

# The unbounded methods pin the bound instead of sweeping it, so that each
# regime gets its own search over the remaining parameters. Their optima are
# not expected to agree: a model that may bend as it likes wants a different
# window length, noise level and uncertainty limit than a constrained one.
_UNBOUNDED_ARGUMENT = "--augmentation.dynamics.lipschitz-constant-upper-bound=None"

METHOD_FIXED_ARGUMENTS: dict[MethodLabel, list[str]] = {
    "tacil-dp-bounded": ["augmentation:tacil"],
    "tacil-dp-unbounded": ["augmentation:tacil", _UNBOUNDED_ARGUMENT],
    "tacil-bc-bounded": ["augmentation:tacil"],
    "tacil-bc-unbounded": ["augmentation:tacil", _UNBOUNDED_ARGUMENT],
    "ccil-bc-noisy": ["augmentation:ccil", "--augmentation.labels.type=noisy_action"],
    "ccil-bc-euler": [
        "augmentation:ccil",
        "--augmentation.labels.type=backward_euler_fast",
    ],
}

_TACIL_SHARED_PARAMETERS: dict[str, Any] = {
    "augmentation.dynamics.window_length": {
        "distribution": "int_uniform",
        "min": 1,
        "max": 15,
    },
    "augmentation.data.action_noise_range_fraction": {
        "distribution": "uniform",
        "min": 0.0,
        "max": 1.0,
    },
    # The uncertainty a single backward step may show, in multiples of the
    # median expert step. The ceiling used to be 5.0, which was too low: no run
    # of 2026-08-15 got past chain length 3, while the runs of 2026-08-12 that
    # scored best ran to length 8. The backward error grows about tenfold every
    # five depths, so reaching those lengths takes a ceiling one to two orders
    # of magnitude above the old one. A ceiling too high only costs a few runs
    # that all end at the run-up cap and look alike; a ceiling too low hides
    # the answer altogether, which is what happened. Should the best runs sit
    # at the ceiling, it is still too low.
    "augmentation.data.uncertainty_limit_step_multiple": {
        "distribution": "log_uniform_values",
        "min": 0.05,
        "max": 500.0,
    },
}

# A range rather than a list of values: a values list is categorical, so the
# search would treat every bound as unrelated to the next and would have to try
# each one before it knows anything. Log scale, because the bound acts on the
# model by its order of magnitude. The ends are the ends of the list that
# 2026-08-12 stepped through, 1 to 100, and the floor is not free: below it the
# network that predicts the state displacement becomes near constant, so every
# ensemble member learns the same function, they never disagree, and an
# uncertainty near zero passes any limit. The first four runs of 2026-08-16
# showed it — the three with a bound under 1 walked all 6177 anchors into the
# run-up cap and cost about thirteen hours each, while the fourth, at 56, ended
# its chains after six steps.
_TACIL_BOUNDED_PARAMETERS: dict[str, Any] = {
    **_TACIL_SHARED_PARAMETERS,
    "augmentation.dynamics.lipschitz_constant_upper_bound": {
        "distribution": "log_uniform_values",
        "min": 1.0,
        "max": 100.0,
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
    "tacil-dp-bounded": _TACIL_BOUNDED_PARAMETERS,
    "tacil-dp-unbounded": _TACIL_SHARED_PARAMETERS,
    "tacil-bc-bounded": _TACIL_BOUNDED_PARAMETERS,
    "tacil-bc-unbounded": _TACIL_SHARED_PARAMETERS,
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


@dataclass(frozen=True)
class Config:
    project_prefix: str
    dataset_percentage: int
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
    project = f"{config.project_prefix}-{config.dataset_percentage}-{config.method}-{condition.label}"

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
            f"--expert-dataset-artifact=datasets/dataset-{config.dataset_percentage}:latest",
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
