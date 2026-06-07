from dataclasses import asdict
from pathlib import Path
from typing import Literal

import torch
import tyro
import wandb
from rich.pretty import pretty_repr
from wandb import Run

from scripts.ashared import create_logger, log_device_info, set_seed
from scripts.evaluation.bryan.evaluation import (
    EvaluationResult,
    PolicyFactory,
    evaluate_policy,
)
from scripts.evaluation.config import Config
from scripts.policies.bc.policy import load_behavior_cloning_policy
from scripts.policies.dp.policy import load_diffusion_policy

logger = create_logger()


def main():
    config = tyro.cli(f=Config, config=[tyro.conf.UsePythonSyntaxForLiteralCollections])
    logger.info(pretty_repr(config))
    _run(config=config)


def _run(config: Config) -> None:
    set_seed(seed=config.seed)
    device = torch.device(device="cuda" if torch.cuda.is_available() else "cpu")
    log_device_info(device=device, logger=logger)

    with wandb.init(
        project="abc", job_type="policy-eval", config=asdict(config)
    ) as run:
        checkpoint_path = (
            Path(
                run.use_artifact(artifact_or_name=config.checkpoint_artifact).download()
            )
            / "checkpoint.pt"
        )
        policy_factory = _build_policy_factory(
            policy_type=config.policy_type,
            checkpoint_path=checkpoint_path,
            device=device,
        )

        clean_result = evaluate_policy(
            policy_factory=policy_factory,
            episode_seeds=config.episode_seeds,
            disturber_config=None,
        )
        disturbed_result = evaluate_policy(
            policy_factory=policy_factory,
            episode_seeds=config.episode_seeds,
            disturber_config=config.disturbance,
        )

        _write_summary(run=run, protocol="clean", result=clean_result)
        _write_summary(run=run, protocol="disturbed", result=disturbed_result)


def _build_policy_factory(
    policy_type: Literal["dp", "bc"], checkpoint_path: Path, device: torch.device
) -> PolicyFactory:
    match policy_type:
        case "dp":
            return load_diffusion_policy(checkpoint_path=checkpoint_path, device=device)
        case "bc":
            return load_behavior_cloning_policy(
                checkpoint_path=checkpoint_path, device=device
            )


def _write_summary(run: Run, protocol: str, result: EvaluationResult) -> None:
    run.summary[f"{protocol}/max_coverage_mean"] = result.max_coverage_mean
    run.summary[f"{protocol}/max_coverage_std"] = result.max_coverage_std
    run.summary[f"{protocol}/success_rate"] = result.success_rate


if __name__ == "__main__":
    main()
