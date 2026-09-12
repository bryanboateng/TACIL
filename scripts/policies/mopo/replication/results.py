"""Puts the reproduced scores next to the published ones.

The published spread matters as much as the published mean. OfflineRL-Kit
averages four seeds, and on some tasks those four seeds are 40 points apart,
so landing far from the mean says nothing there. The verdict column reports
whether a score falls inside one published spread, which is the question a
correctness check actually asks.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from scripts.policies.mopo.replication.tasks import ALL_TASKS
from scripts.shared.log import create_logger
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH

logger = create_logger()

RESULT_DIRECTORY_PATH = OUTPUT_DIRECTORY_PATH / "replication" / "mopo"

_ROW_FORMAT = "{:<32}{:>16}{:>16}{:>10}{:>10}{:>8}"


@dataclass(frozen=True)
class Config:
    result_directory_path: Path = RESULT_DIRECTORY_PATH


def write_result(
    *, task: str, seed: int, score: float, directory_path: Path = RESULT_DIRECTORY_PATH
) -> Path:
    """Records one finished run, so a later table can pick it up."""
    file_path = directory_path / f"{task}-seed{seed}.json"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w") as file:
        json.dump(obj={"task": task, "seed": seed, "score": score}, fp=file, indent=2)
    return file_path


def main() -> None:
    config = tyro.cli(f=Config)
    logger.info(
        "MOPO against OfflineRL-Kit:\n%s",
        format_summary(result_directory_path=config.result_directory_path),
    )


def format_summary(*, result_directory_path: Path = RESULT_DIRECTORY_PATH) -> str:
    scores = _read_scores(directory_path=result_directory_path)

    lines: list[str] = [
        _ROW_FORMAT.format("task", "ours", "published", "gap", "verdict", "seeds")
    ]
    for name, task in ALL_TASKS.items():
        our_scores = scores.get(name)
        if our_scores is None:
            continue
        mean = float(np.mean(our_scores))
        gap = mean - task.published_score
        lines.append(
            _ROW_FORMAT.format(
                name,
                f"{mean:.1f} +- {np.std(our_scores):.1f}",
                f"{task.published_score:.1f} +- {task.published_score_std:.1f}",
                f"{gap:+.1f}",
                "within" if abs(gap) <= task.published_score_std else "outside",
                str(len(our_scores)),
            )
        )
    return "\n".join(lines)


# ---- Private --------------------------------------------


def _read_scores(*, directory_path: Path) -> dict[str, list[float]]:
    scores: dict[str, list[float]] = {}
    for file_path in sorted(directory_path.glob("*.json")):
        with file_path.open() as file:
            record = json.load(fp=file)
        scores.setdefault(record["task"], []).append(float(record["score"]))
    return scores


if __name__ == "__main__":
    main()
