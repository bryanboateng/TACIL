import importlib
from dataclasses import dataclass
from typing import Any, Literal

import tyro
import wandb
from rich.pretty import pretty_repr

from scripts.data.augmentation.tacil.create_data.chain_error import ChainErrorCriterion
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
    # GPI distilled into a policy: its actions, rolled forward through the
    # dynamics model, as training data.
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
    "distill-dp": ["augmentation:distillation"],
    "distill-bc": ["augmentation:distillation"],
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
    # Ceiling on the chain error, in multiples of the median expert step.
    # Which error that is depends on the criterion the sweep was created with,
    # and the scale is not shared between them: an accumulated error grows with
    # depth while a per-step one does not. The range below is wide enough to
    # bracket either, which is why it is not narrowed per criterion. The anchor points: a first step needs roughly the per-step
    # replay error (~0.7 multiples at the accepted-step median), and length 8
    # is estimated between √8 and 8 times that step's error — about 2
    # to 13 multiples — depending on how strongly the members' errors add up.
    # The range brackets that with an order of magnitude on both sides,
    # because the depth curve showed the reward cliff (0.70 at length 8,
    # 0.38 at 14) and the sweep has to see both sides of it.
    "augmentation.data.chain_error_limit_step_multiple": {
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

_DISTILLATION_PARAMETERS: dict[str, Any] = {
    # GPI's own parameters, over the range its own sweep searches: what is
    # being copied is GPI, so how GPI is configured is part of the search.
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
    "augmentation.data.gpi_plan_length": {
        "distribution": "int_uniform",
        "min": 1,
        "max": 32,
    },
    "augmentation.dynamics.window_length": {
        "distribution": "int_uniform",
        "min": 1,
        "max": 15,
    },
    # Same reasoning as for the TACIL bound, see _TACIL_BOUNDED_PARAMETERS.
    "augmentation.dynamics.lipschitz_constant_upper_bound": {
        "distribution": "log_uniform_values",
        "min": 1.0,
        "max": 120.0,
    },
    # How hard the block is shoved before GPI has to bring it back.
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


# Sizes as strings, because a sweep value has to survive the round trip
# through the command line; tyro reads them back with Python collection syntax.
# The reference size is the largest here: at 5% of the data it has more
# parameters than there are transitions to fit them on.

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
    "distill-dp": _DISTILLATION_PARAMETERS,
    "distill-bc": _DISTILLATION_PARAMETERS,
}


@dataclass(frozen=True)
class Config:
    project_prefix: str
    dataset_percentage: int
    method: MethodLabel

    seed: int

    # Which stopping rule a TACIL sweep searches under. One sweep per rule
    # rather than a rule drawn per run: the search would otherwise spend its
    # budget unevenly across the rules and none of them would be settled.
    # It lands in the project name, so the sweeps stay apart.
    chain_error_criterion: ChainErrorCriterion | None = None

    def __post_init__(self) -> None:
        if self.chain_error_criterion is not None and not self.method.startswith(
            "tacil"
        ):
            raise ValueError(
                f"A chain error criterion only applies to the TACIL methods, "
                f"which are the ones that walk a chain; the method is "
                f"{self.method}."
            )

    sweep_method: SearchLabel = "bayes"

    entity: str = "bryanboateng-team"


def main() -> None:
    config = tyro.cli(f=Config, config=[tyro.conf.UsePythonSyntaxForLiteralCollections])
    logger.info(pretty_repr(config))

    project = f"{config.project_prefix}-{config.dataset_percentage}-{config.method}"
    if config.chain_error_criterion is not None:
        project += f"-{config.chain_error_criterion.replace('_', '-')}"

    sweep_configuration = _build_sweep_configuration(config=config, project=project)
    logger.info(pretty_repr(sweep_configuration))

    _verify_command_parses(
        sweep_configuration=sweep_configuration, method=config.method
    )

    sweep_id = wandb.sweep(
        sweep=sweep_configuration, entity=config.entity, project=project
    )

    logger.info("Sweep path: %s", f"{config.entity}/{project}/{sweep_id}")


def _verify_command_parses(
    *, sweep_configuration: dict[str, Any], method: MethodLabel
) -> None:
    """Parse the command this sweep would run, with one sample value per
    parameter.

    A wrongly ordered command does not fail loudly: the run dies inside the
    argument parser, before wandb.init, and all that remains is a crashed run
    with an empty summary. Two sweeps have already been lost that way, so the
    command is parsed here, while it can still be fixed.
    """
    sample_arguments = [
        f"--{name}={_sample_value(specification=specification)}"
        for name, specification in sweep_configuration["parameters"].items()
    ]

    # The first four entries are the interpreter and the module to run.
    arguments: list[str] = []
    for token in sweep_configuration["command"][4:]:
        if token == "${args_no_boolean_flags}":
            arguments.extend(sample_arguments)
        else:
            arguments.append(token)

    run_module = importlib.import_module(name=METHOD_MODULES[method])
    try:
        tyro.cli(
            f=run_module.Config,
            args=arguments,
            config=[tyro.conf.UsePythonSyntaxForLiteralCollections],
        )
    except SystemExit as exit_error:
        raise RuntimeError(
            f"The command for {method} does not parse, so every run of this "
            f"sweep would die before it could log anything. Arguments:\n  "
            + "\n  ".join(arguments)
        ) from exit_error


def _sample_value(*, specification: dict[str, Any]) -> Any:
    if "values" in specification:
        return specification["values"][0]
    return specification["min"]


def _build_sweep_configuration(*, config: Config, project: str) -> dict[str, Any]:
    pipeline_module = METHOD_MODULES[config.method]
    parameters = METHOD_PARAMETERS[config.method]

    # Where the fixed arguments go depends on what is being searched. A
    # subcommand such as `augmentation:tacil` closes the parent parser: options
    # of the run itself are only read before it, options of the augmentation
    # only after it. So a sweep over `augmentation.*` needs the subcommand
    # first, and a sweep over `policy.*` needs it last. Getting this backwards
    # does not fail loudly -- the run dies while parsing, before wandb.init, so
    # it shows up only as a crashed run with nothing logged.
    fixed_arguments = list(METHOD_FIXED_ARGUMENTS[config.method])
    if config.chain_error_criterion is not None:
        fixed_arguments.append(
            f"--augmentation.data.chain-error-criterion={config.chain_error_criterion}"
        )
    swept_arguments = "${args_no_boolean_flags}"
    method_arguments = (
        [swept_arguments, *fixed_arguments]
        if all(key.startswith("policy.") for key in parameters)
        else [*fixed_arguments, swept_arguments]
    )

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
            *method_arguments,
        ],
        "parameters": parameters,
    }


if __name__ == "__main__":
    main()
