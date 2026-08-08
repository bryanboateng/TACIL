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
from scripts.evaluation.evaluate_policy import STAGE_NAME
from scripts.shared.log import create_logger

logger = create_logger()

DatasetLabel = Literal["full", "sub025"]

SearchLabel = Literal["bayes", "random"]

# The two CCIL label generators are separate methods: a shared sweep would
# spend its budget exploring the action noise in trials that ignore it.
# The unaugmented baselines are not here: with the policy hyperparameters
# fixed they have nothing to search, so they are single runs of
# run_dp_pipeline and run_bc_pipeline rather than sweeps.
MethodLabel = Literal["tacil-dp", "tacil-bc", "ccil-bc-noisy", "ccil-bc-euler", "gpi"]

# The policy module each method's trials run. The augmentation is part of the
# policy's own command line, so a method is a module plus a subcommand.
METHOD_MODULES: dict[MethodLabel, str] = {
    "tacil-dp": "scripts.policies.dp.run",
    "tacil-bc": "scripts.policies.bc.run",
    "ccil-bc-noisy": "scripts.policies.bc.run",
    "ccil-bc-euler": "scripts.policies.bc.run",
    "gpi": "scripts.policies.gpi.run",
}

# Fixed per-method arguments, appended to the command before the swept ones.
# The subcommand token must come before the swept options it configures.
METHOD_FIXED_ARGUMENTS: dict[MethodLabel, list[str]] = {
    "tacil-dp": ["augmentation:tacil"],
    "tacil-bc": ["augmentation:tacil"],
    "ccil-bc-noisy": ["augmentation:ccil", "--augmentation.labels.type=noisy_action"],
    "ccil-bc-euler": [
        "augmentation:ccil",
        "--augmentation.labels.type=backward_euler_fast",
    ],
}

# Every swept parameter configures the augmentation, so the keys carry its
# prefix and the stage it belongs to: wandb turns each into
# --augmentation.<stage>.<name>=<value>.
_TACIL_PARAMETERS: dict[str, Any] = {
    # A ceiling on how steep the dynamics model may be. None is a real
    # option, not a stand-in number: the model is then built without the
    # bounding layer at all, which the held-out chains say contact needs.
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
    "augmentation.data.error_limit_fraction": {
        "distribution": "log_uniform_values",
        "min": 0.125,
        "max": 32.0,
    },
    "augmentation.data.disagreement_weight": {
        "distribution": "uniform",
        "min": 0.0,
        "max": 1.0,
    },
}

# Their configs pick one Lipschitz variant per environment; here the sweep
# chooses. The two "slack" variants are not ported (second network).
_CCIL_SHARED_PARAMETERS: dict[str, Any] = {
    "augmentation.dynamics.lipschitz_type": {
        "values": ["soft_sampling", "spectral_normalization", "none"]
    },
    # Values around their configs (they use 1.0 to 4.0).
    "augmentation.dynamics.lipschitz_constraint": {
        "values": [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    },
    # The paper's rejection quantile (its Fig. 5).
    "augmentation.labels.rejection_quantile": {
        "distribution": "uniform",
        "min": 0.0,
        "max": 1.0,
    },
}

# What each method's sweep searches.
METHOD_PARAMETERS: dict[MethodLabel, dict[str, Any]] = {
    "tacil-dp": _TACIL_PARAMETERS,
    "tacil-bc": _TACIL_PARAMETERS,
    "ccil-bc-noisy": {
        **_CCIL_SHARED_PARAMETERS,
        # Their delta, absolute in raw action units (their configs use 1e-5;
        # PushT actions are pixels, so the sweep spans up to pixel scale).
        "augmentation.labels.action_noise_std": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 10.0,
        },
    },
    "ccil-bc-euler": _CCIL_SHARED_PARAMETERS,
    "gpi": {
        "policy.guidance.lambda1": {
            "distribution": "log_uniform_values",
            "min": 0.1,
            "max": 10.0,
        },
        "policy.action_horizon": {"distribution": "int_uniform", "min": 1, "max": 16},
    },
}

# The datasets a sweep can start from, under the label that names the project.
EXPERT_DATASET_ARTIFACTS: dict[DatasetLabel, str] = {
    "full": "bryanboateng-team/datasets/expert-dataset:v0",
    "sub025": "bryanboateng-team/datasets/subsampled-dataset-0.25:v0",
}


@dataclass(frozen=True)
class Config:
    # Names the project, together with the dataset, method, and condition.
    project_prefix: str
    dataset: DatasetLabel
    method: MethodLabel
    floor_condition: FloorCondition
    disturbance_level: DisturbanceLevel | None

    # Every trial in the sweep runs with this seed, so the trials differ only
    # in the parameters being searched. Pick the seeds for the final runs
    # separately, and not this one: the winning parameters were chosen on it.
    seed: int

    # How the next trial is chosen. Bayesian search models the reward from the
    # finished trials and proposes where it looks best, which pays off in a
    # small space. Random search ignores the finished trials: in a space with
    # many parameters it takes a long time before the model is worth
    # anything, and random draws stay independent no matter how many agents
    # run at once.
    search: SearchLabel = "bayes"

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
    pipeline_module = METHOD_MODULES[config.method]
    return {
        "program": pipeline_module.replace(".", "/") + ".py",
        "method": config.search,
        # Written by the evaluation stage, which owns the prefix.
        "metric": {
            "name": f"{STAGE_NAME}/{condition.label}/reward_mean",
            "goal": "maximize",
        },
        # The pipeline is a module, not a script: it imports from the repository
        # root, which is only on the path when python is started with -m.
        "command": [
            "${env}",
            "${interpreter}",
            "-m",
            pipeline_module,
            f"--wandb-project={project}",
            f"--expert-dataset-artifact={EXPERT_DATASET_ARTIFACTS[config.dataset]}",
            f"--floor-condition={config.floor_condition.name}",
            f"--disturbance-level={_disturbance_level_argument(config=config)}",
            f"--seed={config.seed}",
            *METHOD_FIXED_ARGUMENTS.get(config.method, []),
            # Booleans arrive as flags (present or absent) rather than
            # --flag=True, which is what the pipeline's command line takes.
            "${args_no_boolean_flags}",
        ],
        "parameters": METHOD_PARAMETERS[config.method],
    }


def _disturbance_level_argument(*, config: Config) -> str:
    level = config.disturbance_level
    return "None" if level is None else level.name


if __name__ == "__main__":
    main()
