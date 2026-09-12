from __future__ import annotations

import argparse
import logging
import os
import pickle
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

import numpy as np
import torch
import yaml

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(name=__name__)

EPISODE_COUNT = 100

POLICY_NAME_PATTERN = re.compile(r"^(?P<variant>[a-z]+)-seed(?P<seed>\d+)\.pt$")


@dataclass(frozen=True)
class Policy:
    task: str
    seed: int
    file_path: Path

    @property
    def rewards_file_path(self) -> Path:
        return self.file_path.with_suffix(".pkl")


class _EnvironmentConfig:
    def __init__(self, environment_name: str) -> None:
        self.env = environment_name


def main() -> None:
    arguments = _parse_arguments()
    ccil_directory_path = Path(arguments.ccil_directory).expanduser()
    sys.path.insert(0, str(ccil_directory_path / "correct_il"))

    if arguments.policy is not None:
        policies = [
            Policy(
                task=arguments.task,
                seed=arguments.seed,
                file_path=Path(arguments.policy).expanduser(),
            )
        ]
    else:
        policies = list(
            _find_policies(directory_path=Path(arguments.policy_directory).expanduser())
        )

    for policy in policies:
        if policy.rewards_file_path.is_file():
            continue
        rewards = _score(
            ccil_directory_path=ccil_directory_path,
            policy=policy,
            episode_count=arguments.episodes,
        )
        with policy.rewards_file_path.open("wb") as file:
            pickle.dump(rewards, file)


# ---- Private --------------------------------------------


def _find_policies(*, directory_path: Path) -> Iterator[Policy]:
    for file_path in sorted(directory_path.glob("*/*.pt")):
        match = POLICY_NAME_PATTERN.match(file_path.name)
        if match is None:
            continue
        yield Policy(
            task=file_path.parent.name,
            seed=int(match.group("seed")),
            file_path=file_path,
        )


def _score(*, ccil_directory_path: Path, policy: Policy, episode_count: int) -> list:
    from utils import (  # ty: ignore[unresolved-import]
        D3Agent,
        evaluate_on_environment,
        load_env,
    )

    with (ccil_directory_path / "config" / f"{policy.task}.yml").open() as file:
        task_config = yaml.safe_load(file)
    noise = (task_config["eval"]["noise"] or [0])[0]

    random.seed(policy.seed)
    np.random.seed(policy.seed)
    torch.manual_seed(policy.seed)

    environment, is_metaworld = load_env(_EnvironmentConfig(task_config["env"]))
    environment.seed(policy.seed)
    environment.action_space.seed(policy.seed)
    environment.observation_space.seed(policy.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy_model = torch.jit.load(str(policy.file_path))
    policy_model.to(device)

    logger.info(
        "Scoring %s in %s, seed %d, evaluation noise %s.",
        policy.file_path.name,
        task_config["env"],
        policy.seed,
        noise,
    )
    rewards, success_count = evaluate_on_environment(
        environment,
        D3Agent(policy_model, device),
        n_trials=episode_count,
        metaworld=is_metaworld,
        sensor_noise_size=noise,
        actuator_noise_size=noise,
    )
    environment.close()

    logger.info(
        "Reward %.2f +- %.2f over %d episodes, success rate %.2f.",
        float(np.mean(rewards)),
        float(np.std(rewards)),
        len(rewards),
        success_count / float(episode_count),
    )
    return rewards


def _parse_arguments() -> Any:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ccil-directory", required=True)
    parser.add_argument(
        "--policy-directory", default=None, help="a tree of <task>/<variant>-seed<n>.pt"
    )
    parser.add_argument("--policy", default=None, help="a single TorchScript .pt file")
    parser.add_argument("--task", default=None, help="a config name, e.g. hopper")
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--episodes", type=int, default=EPISODE_COUNT)
    arguments = parser.parse_args()

    if arguments.policy_directory is None and arguments.policy is None:
        parser.error("pass --policy-directory or --policy")
    if arguments.policy_directory is not None and arguments.policy is not None:
        parser.error("pass --policy-directory or --policy, not both")
    if arguments.policy is not None and arguments.task is None:
        parser.error("--policy needs --task")
    return arguments


if __name__ == "__main__":
    main()
