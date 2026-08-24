from dataclasses import dataclass

import tyro
import wandb

from scripts.evaluation.core.condition import VALIDATION_CONDITION
from scripts.evaluation.evaluate_policy import STAGE_NAME
from scripts.shared.log import create_logger
from scripts.sweep.create_sweep import (
    METHOD_FIXED_ARGUMENTS,
    METHOD_MODULES,
    METHOD_PARAMETERS,
    MethodLabel,
)

logger = create_logger()

# The sweep evaluates one condition only, so that is the score the search
# optimised and the score we rank the finished runs by. The final runs then
# re-run the winners on every condition (see `--evaluate-all-conditions`).
RANKING_METRIC = f"{STAGE_NAME}/{VALIDATION_CONDITION.label}/reward_mean"


@dataclass(frozen=True)
class Config:
    dataset_percentage: int
    method: MethodLabel

    sweep_id: str

    project_prefix: str = "2026-08-22"
    final_project_prefix: str = "2026-08-24-final"
    entity: str = "bryanboateng-team"

    top_k: int = 3
    seeds: tuple[int, ...] = (0, 1, 2)


def main() -> None:
    config = tyro.cli(f=Config, config=[tyro.conf.UsePythonSyntaxForLiteralCollections])

    sweep_project = (
        f"{config.project_prefix}-{config.dataset_percentage}-{config.method}"
    )
    sweep = wandb.Api().sweep(
        f"{config.entity}/{sweep_project}/{config.sweep_id}"
    )

    ranked = sorted(
        (run for run in sweep.runs if RANKING_METRIC in run.summary),
        key=lambda run: run.summary[RANKING_METRIC],
        reverse=True,
    )
    if len(ranked) < config.top_k:
        logger.warning(
            "Sweep %s has only %d scored runs, fewer than the requested top %d.",
            sweep_project,
            len(ranked),
            config.top_k,
        )

    final_project = (
        f"{config.final_project_prefix}-{config.dataset_percentage}-{config.method}"
    )
    module = METHOD_MODULES[config.method]

    for run in ranked[: config.top_k]:
        swept_arguments = [
            f"--{key}={run.config[key]}" for key in METHOD_PARAMETERS[config.method]
        ]
        for seed in config.seeds:
            command = [
                "python",
                "-m",
                module,
                f"--wandb-project={final_project}",
                f"--expert-dataset-artifact=datasets/dataset-{config.dataset_percentage}:latest",
                f"--seed={seed}",
                *METHOD_FIXED_ARGUMENTS[config.method],
                *swept_arguments,
                "--evaluate-all-conditions",
            ]
            print(" ".join(command))


if __name__ == "__main__":
    main()
