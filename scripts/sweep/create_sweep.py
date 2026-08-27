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
    # GPI distilled into a policy: its actions, rolled forward through the
    # dynamics model, as training data. The decisive control -- it says whether
    # a learned policy can reach GPI's score at all on this data.
    "distill-dp",
    "distill-bc",
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
    "distill-dp": "scripts.policies.dp.run",
    "distill-bc": "scripts.policies.bc.run",
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
    "distill-dp": ["augmentation:recovery", "--augmentation.data.action-source=gpi"],
    "distill-bc": ["augmentation:recovery", "--augmentation.data.action-source=gpi"],
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
    # Lookahead of the planner. 2 already beats greedy; 12 is about as deep as
    # a rollout of the one-step model stays worth judging on.
    #
    # The candidate step size is not searched next to it any more. Since the
    # search starts at GPI's action, that number is a correction radius around
    # an action that is already expert-shaped, not the scale of a search from
    # nothing, and its default is small on purpose.
    "augmentation.data.plan_horizon": {
        "distribution": "int_uniform",
        "min": 2,
        "max": 12,
    },
    # Ceiling on the chain length. It is the depth lever: the model's forward
    # error grows with every step it predicts on its own, so a long chain ends
    # on states the simulator could never produce, and the policy spends its
    # capacity on them. TACIL's depth curve showed the same trade-off.
    #
    # It overlaps with the replay error limit below, which measures that same
    # unreliability directly instead of by step count. Both stay, as they did
    # in TACIL: the limit passes a chain the members agree on, and this cap
    # still refuses one they only agree on because they share a blind spot.
    #
    # The arrival tolerance and the target count are not searched beside it.
    # A tolerance of one median expert step is what "back at a demonstrated
    # state" means rather than a value to tune, and aiming at the 32 nearest
    # states is about having somewhere to go, not a sensitive quantity. With
    # roughly 80 runs per sweep, every extra axis thins the search, and these
    # two would have thinned it without adding a distinct lever.
    "augmentation.data.max_chain_length": {
        "distribution": "int_uniform",
        "min": 2,
        "max": 16,
    },
    # Same role as in the TACIL sweeps, but the floor is raised. Under one
    # median expert step the test rejected every chain a run produced, so the
    # run died with no data: half of 2026-08-27d's runs were lost that way,
    # and none of them could have scored.
    "augmentation.data.replay_error_limit_step_multiple": {
        "distribution": "log_uniform_values",
        "min": 1.0,
        "max": 120.0,
    },
}

_GPI_WEIGHT_PARAMETERS: dict[str, Any] = {
    "augmentation.data.gpi_progression_weight": {
        "distribution": "log_uniform_values",
        "min": 0.1,
        "max": 10.0,
    },
    "augmentation.data.gpi_attraction_weight": {
        "distribution": "log_uniform_values",
        "min": 0.1,
        "max": 10.0,
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

_DISTILLATION_PARAMETERS: dict[str, Any] = {
    **_GPI_WEIGHT_PARAMETERS,
    "augmentation.dynamics.window_length": {
        "distribution": "int_uniform",
        "min": 1,
        "max": 15,
    },
    "augmentation.dynamics.lipschitz_constant_upper_bound": {
        "distribution": "log_uniform_values",
        "min": 1.0,
        "max": 120.0,
    },
    "augmentation.data.disturbance_range_fraction": {
        "distribution": "log_uniform_values",
        "min": 0.02,
        "max": 0.5,
    },
    # How many steps GPI is followed through the model. Every step adds model
    # error, so this trades coverage of GPI's behaviour against how real the
    # states it is demonstrated on still are.
    "augmentation.data.max_chain_length": {
        "distribution": "int_uniform",
        "min": 1,
        "max": 16,
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
    "recovery-dp-bounded": {**_RECOVERY_BOUNDED_PARAMETERS, **_GPI_WEIGHT_PARAMETERS},
    "recovery-dp-unbounded": {**_RECOVERY_SHARED_PARAMETERS, **_GPI_WEIGHT_PARAMETERS},
    "recovery-bc-bounded": {**_RECOVERY_BOUNDED_PARAMETERS, **_GPI_WEIGHT_PARAMETERS},
    "recovery-bc-unbounded": {**_RECOVERY_SHARED_PARAMETERS, **_GPI_WEIGHT_PARAMETERS},
    # No planner, so none of its parameters apply: what shapes the data is how
    # far the block is shoved, how long GPI is followed, and GPI's own weights.
    "distill-dp": _DISTILLATION_PARAMETERS,
    "distill-bc": _DISTILLATION_PARAMETERS,
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
