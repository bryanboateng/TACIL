from dataclasses import dataclass
from typing import Any, Literal

import tyro
import wandb
from rich.pretty import pretty_repr

from scripts.evaluation.core.condition import VALIDATION_CONDITION
from scripts.evaluation.evaluate_policy import STAGE_NAME
from scripts.shared.log import create_logger

logger = create_logger()

SearchLabel = Literal["bayes", "random"]

MethodLabel = Literal[
    "tacil-dp-bounded",
    "tacil-dp-unbounded",
    "tacil-bc-bounded",
    "tacil-bc-unbounded",
    "ccil-bc",
    "gpi",
    # Model-predictive recovery: disturb the block at each anchor, plan the
    # action chain back with the dynamics ensemble, keep only chains whose
    # member replay agrees they arrive.
    "recovery-dp-bounded",
    "recovery-dp-unbounded",
    "recovery-bc-bounded",
    "recovery-bc-unbounded",
]

METHOD_MODULES: dict[MethodLabel, str] = {
    "tacil-dp-bounded": "scripts.policies.dp.run",
    "tacil-dp-unbounded": "scripts.policies.dp.run",
    "tacil-bc-bounded": "scripts.policies.bc.run",
    "tacil-bc-unbounded": "scripts.policies.bc.run",
    "ccil-bc": "scripts.policies.bc.run",
    "gpi": "scripts.policies.gpi.run",
    "recovery-dp-bounded": "scripts.policies.dp.run",
    "recovery-dp-unbounded": "scripts.policies.dp.run",
    "recovery-bc-bounded": "scripts.policies.bc.run",
    "recovery-bc-unbounded": "scripts.policies.bc.run",
}

# The unbounded methods pin the bound instead of sweeping it, so that each
# regime gets its own search over the remaining parameters. Their optima are
# not expected to agree: a model that may bend as it likes wants a different
# window length, noise level and replay error limit than a constrained one.
_UNBOUNDED_ARGUMENT = "--augmentation.dynamics.lipschitz-constant-upper-bound=None"

METHOD_FIXED_ARGUMENTS: dict[MethodLabel, list[str]] = {
    "tacil-dp-bounded": ["augmentation:tacil"],
    "tacil-dp-unbounded": ["augmentation:tacil", _UNBOUNDED_ARGUMENT],
    "tacil-bc-bounded": ["augmentation:tacil"],
    "tacil-bc-unbounded": ["augmentation:tacil", _UNBOUNDED_ARGUMENT],
    "ccil-bc": ["augmentation:ccil"],
    # GPI has no augmentation stage, but the field is still part of the run
    # config, and tyro asks for every field.
    "gpi": ["--augmentation=None"],
    "recovery-dp-bounded": ["augmentation:recovery"],
    "recovery-dp-unbounded": ["augmentation:recovery", _UNBOUNDED_ARGUMENT],
    "recovery-bc-bounded": ["augmentation:recovery"],
    "recovery-bc-unbounded": ["augmentation:recovery", _UNBOUNDED_ARGUMENT],
}

_TACIL_SHARED_PARAMETERS: dict[str, Any] = {
    "augmentation.dynamics.window_length": {
        "distribution": "int_uniform",
        "min": 1,
        "max": 15,
    },
    # Log scale, not uniform: under uniform 0..1 only 7% of draws land below
    # 0.07, yet that is where every good run of 2026-08-12 sat (best at 0.017,
    # eval reward against noise ρ = −0.53). The uniform range made the
    # 2026-08-16 sweep spend 93% of its runs where none had ever scored, which
    # is why it found nothing. This range puts half the draws below 0.017 and
    # still reaches far above every noise level that ever worked.
    "augmentation.data.action_noise_range_fraction": {
        "distribution": "log_uniform_values",
        "min": 0.001,
        "max": 0.3,
    },
    # Ceiling on the replay error: how far the members' replay
    # endpoints may land from the anchor, in multiples of the median expert
    # step. Old per-step limits do not transfer, because this error accumulates
    # over the chain. The anchor points: a first step needs roughly the per-step
    # replay error (~0.7 multiples at the accepted-step median), and length 8
    # is estimated between √8 and 8 times that step's error — about 2
    # to 13 multiples — depending on how strongly the members' errors add up.
    # The range brackets that with an order of magnitude on both sides,
    # because the depth curve showed the reward cliff (0.70 at length 8,
    # 0.38 at 14) and the sweep has to see both sides of it.
    "augmentation.data.replay_error_limit_step_multiple": {
        "distribution": "log_uniform_values",
        "min": 0.05,
        "max": 120.0,
    },
}

# A range rather than a list of values: a values list is categorical, so the
# search would treat every bound as unrelated to the next and would have to try
# each one before it knows anything. Log scale, because the bound acts on the
# model by its order of magnitude. The ends are the ends of the list that
# 2026-08-12 stepped through, 1 to 100, and the floor is not free: below it the
# network that predicts the state displacement becomes near constant, so every
# ensemble member learns the same function, they never disagree, and an
# replay error near zero passes any limit. The first four runs of 2026-08-16
# showed it — the three with a bound under 1 walked all 6177 anchors into the
# run-up cap and cost about thirteen hours each, while the fourth, at 56, ended
# its chains after six steps.
_TACIL_BOUNDED_PARAMETERS: dict[str, Any] = {
    **_TACIL_SHARED_PARAMETERS,
    "augmentation.dynamics.lipschitz_constant_upper_bound": {
        "distribution": "log_uniform_values",
        "min": 1.0,
        "max": 120.0,
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

_RECOVERY_SHARED_PARAMETERS: dict[str, Any] = {
    "augmentation.dynamics.window_length": {
        "distribution": "int_uniform",
        "min": 1,
        "max": 15,
    },
    # How hard the block is shoved before the planner has to bring it back.
    # The low end is barely more than an expert step; the high end throws the
    # block across half its travelled range, which is a harder recovery than
    # the evaluation disturbances create.
    "augmentation.data.disturbance_range_fraction": {
        "distribution": "log_uniform_values",
        "min": 0.02,
        "max": 0.5,
    },
    # Step size of the candidate action walks. Expert actions move a few
    # thousandths to a few hundredths of the range per step; the span covers
    # from below that (careful, precise pushes) to far above (fast repositioning).
    "augmentation.data.plan_step_range_fraction": {
        "distribution": "log_uniform_values",
        "min": 0.005,
        "max": 0.2,
    },
    # Lookahead of the planner. 2 already beats greedy; 12 is about as deep as
    # a rollout of the one-step model stays worth judging on.
    "augmentation.data.plan_horizon": {
        "distribution": "int_uniform",
        "min": 2,
        "max": 12,
    },
    # Same role and range as in the TACIL sweeps: the error is measured with
    # the same replay test, so the scale transfers.
    "augmentation.data.replay_error_limit_step_multiple": {
        "distribution": "log_uniform_values",
        "min": 0.05,
        "max": 120.0,
    },
}

_RECOVERY_BOUNDED_PARAMETERS: dict[str, Any] = {
    **_RECOVERY_SHARED_PARAMETERS,
    # Same reasoning as for the TACIL bound, see _TACIL_BOUNDED_PARAMETERS.
    "augmentation.dynamics.lipschitz_constant_upper_bound": {
        "distribution": "log_uniform_values",
        "min": 1.0,
        "max": 120.0,
    },
}

METHOD_PARAMETERS: dict[MethodLabel, dict[str, Any]] = {
    "tacil-dp-bounded": _TACIL_BOUNDED_PARAMETERS,
    "tacil-dp-unbounded": _TACIL_SHARED_PARAMETERS,
    "tacil-bc-bounded": _TACIL_BOUNDED_PARAMETERS,
    "tacil-bc-unbounded": _TACIL_SHARED_PARAMETERS,
    "ccil-bc": {
        **_CCIL_SHARED_PARAMETERS,
        "augmentation.labels.type": {"values": ["noisy_action", "backward_euler_fast"]},
        "augmentation.labels.action_noise_std": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 10.0,
        },
    },
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
        "policy.plan_length": {"distribution": "int_uniform", "min": 1, "max": 32},
    },
    "recovery-dp-bounded": _RECOVERY_BOUNDED_PARAMETERS,
    "recovery-dp-unbounded": _RECOVERY_SHARED_PARAMETERS,
    "recovery-bc-bounded": _RECOVERY_BOUNDED_PARAMETERS,
    "recovery-bc-unbounded": _RECOVERY_SHARED_PARAMETERS,
}


@dataclass(frozen=True)
class Config:
    project_prefix: str
    dataset_percentage: int
    method: MethodLabel

    seed: int

    sweep_method: SearchLabel = "bayes"

    entity: str = "bryanboateng-team"


def main() -> None:
    config = tyro.cli(f=Config, config=[tyro.conf.UsePythonSyntaxForLiteralCollections])
    logger.info(pretty_repr(config))

    project = f"{config.project_prefix}-{config.dataset_percentage}-{config.method}"

    sweep_configuration = _build_sweep_configuration(config=config, project=project)
    logger.info(pretty_repr(sweep_configuration))

    sweep_id = wandb.sweep(
        sweep=sweep_configuration, entity=config.entity, project=project
    )

    logger.info("Sweep path: %s", f"{config.entity}/{project}/{sweep_id}")


def _build_sweep_configuration(*, config: Config, project: str) -> dict[str, Any]:
    pipeline_module = METHOD_MODULES[config.method]
    return {
        "program": pipeline_module.replace(".", "/") + ".py",
        "method": config.sweep_method,
        "metric": {
            "name": f"{STAGE_NAME}/{VALIDATION_CONDITION.label}/reward_mean",
            "goal": "maximize",
        },
        "command": [
            "${env}",
            "${interpreter}",
            "-m",
            pipeline_module,
            f"--wandb-project={project}",
            f"--expert-dataset-artifact=datasets/dataset-{config.dataset_percentage}:latest",
            f"--seed={config.seed}",
            # Indexed, not `.get`: a method missing here is a mistake, and a
            # silent empty default hides it until the sweep agent fails.
            *METHOD_FIXED_ARGUMENTS[config.method],
            "${args_no_boolean_flags}",
        ],
        "parameters": METHOD_PARAMETERS[config.method],
    }


if __name__ == "__main__":
    main()
