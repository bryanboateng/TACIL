from dataclasses import dataclass

import tyro
import wandb

from scripts.evaluation.evaluate_policy import ALL_CONDITIONS_LABEL, STAGE_NAME
from scripts.shared.log import create_logger
from scripts.sweep.create_sweep import (
    METHOD_FIXED_ARGUMENTS,
    METHOD_MODULES,
    METHOD_PARAMETERS,
    MethodLabel,
)

logger = create_logger()

RANKING_METRIC = f"{STAGE_NAME}/{ALL_CONDITIONS_LABEL}/1_reward_mean"


@dataclass(frozen=True)
class Config:
    dataset_percentage: int
    method: MethodLabel

    sweep_id: str

    # Both name a project rather than defaulting to one: a stale default would
    # read an old sweep and print its winners without saying so.
    project_prefix: str
    final_project: str

    top_k: int = 3
    seeds: tuple[int, ...] = (0, 1, 2)


def main() -> None:
    config = tyro.cli(f=Config, config=[tyro.conf.UsePythonSyntaxForLiteralCollections])

    sweep_project = (
        f"{config.project_prefix}-{config.dataset_percentage}-{config.method}"
    )
    sweep = wandb.Api().sweep(f"{sweep_project}/{config.sweep_id}")

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
                f"--wandb-project={config.final_project}",
                f"--expert-dataset-artifact=datasets/dataset-{config.dataset_percentage}:latest",
                f"--seed={seed}",
                # Top-level option: it must precede the augmentation subcommand,
                # or tyro reads it as a (missing) option of that subcommand.
                *METHOD_FIXED_ARGUMENTS[config.method],
                *swept_arguments,
            ]
            print(" ".join(command))


if __name__ == "__main__":
    main()
