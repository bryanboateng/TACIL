import subprocess
from dataclasses import dataclass
from pathlib import Path

import tyro
from rich.pretty import pretty_repr

from scripts.data.augmentation.ccil.replication import results, training
from scripts.shared.log import create_logger
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH

logger = create_logger()


@dataclass(frozen=True)
class Config:
    ccil_directory_path: Path
    wandb_project: str

    output_directory_path: Path = OUTPUT_DIRECTORY_PATH / "replication"

    # The tasks that mujoco.sh and metaworld.sh cover..
    tasks: tuple[str, ...] = (
        "hopper",
        "walker2d",
        "halfcheetah",
        "ant",
        "coffee_pull",
        "coffee_push",
        "button",
        "drawer_close",
    )

    seeds: tuple[int, ...] = tuple(range(40, 50))

    def __post_init__(self) -> None:
        if len(self.tasks) == 0:
            raise ValueError("tasks must name at least one task, but it is empty.")
        if len(self.seeds) == 0:
            raise ValueError("seeds must name at least one seed, but it is empty.")


def main() -> None:
    config = tyro.cli(f=Config, config=[tyro.conf.UsePythonSyntaxForLiteralCollections])
    logger.info(pretty_repr(config))
    _run(config=config)


# ---- Private --------------------------------------------


def _run(*, config: Config) -> None:
    task_config_paths = {
        task: config.ccil_directory_path / "config" / f"{task}.yml"
        for task in config.tasks
    }
    for task_config_path in task_config_paths.values():
        if not task_config_path.is_file():
            raise FileNotFoundError(f"No CCIL config at {task_config_path}")

    for task in config.tasks:
        for seed in config.seeds:
            for label, augmentation in _augmentations(task=task):
                policy_file_path = (
                    config.output_directory_path / task / f"{label}-seed{seed}.pt"
                )
                if policy_file_path.is_file():
                    logger.info("Skipping %s, it already exists.", policy_file_path)
                    continue

                logger.info("Training %s.", policy_file_path)
                training.train(
                    config=training.Config(
                        seed=seed,
                        wandb_project=config.wandb_project,
                        task_config_path=task_config_paths[task],
                        policy_file_path=policy_file_path,
                        augmentation=augmentation,
                    )
                )

    _evaluate(config=config)


def _evaluate(*, config: Config) -> None:
    # evaluate.py needs CCIL's gym, d4rl and mujoco-py, which only their
    # Python 3.8 venv has. mujoco-py also needs LD_LIBRARY_PATH, and that is
    # inherited from the calling shell.
    interpreter_path = config.ccil_directory_path / ".venv" / "bin" / "python"
    if not interpreter_path.is_file():
        raise FileNotFoundError(f"No CCIL interpreter at {interpreter_path}")

    command = (
        str(interpreter_path),
        str(Path(__file__).with_name("evaluate.py")),
        "--ccil-directory",
        str(config.ccil_directory_path),
        "--policy-directory",
        str(config.output_directory_path),
    )
    logger.info("Scoring the policies: %s", " ".join(command))
    subprocess.run(command, check=True)

    logger.info(
        "Replication against the CCIL paper:\n%s",
        results.format_summary(directory_path=config.output_directory_path),
    )


def _augmentations(
    *, task: str
) -> tuple[tuple[str, training.Augmentation | None], ...]:
    noise_size = 1.0 if task == "ant" else 1e-4
    return (
        ("none", None),
        ("noise", training.Noise(size=noise_size)),
        ("ccil", training.Ccil()),
    )


if __name__ == "__main__":
    main()
