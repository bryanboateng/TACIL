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

_ROW_FORMAT = "{:<30}{:>16}{:>16}{:>16}{:>10}{:>10}{:>7}"


@dataclass(frozen=True)
class PaperResult:
    mean: float
    standard_deviation: float


PAPER_RESULTS = {
    "halfcheetah-medium-v2": PaperResult(42.3, 1.6),
    "hopper-medium-v2": PaperResult(28.0, 12.4),
    "walker2d-medium-v2": PaperResult(17.8, 19.3),
    "halfcheetah-medium-replay-v2": PaperResult(53.1, 2.0),
    "hopper-medium-replay-v2": PaperResult(67.5, 24.7),
    "walker2d-medium-replay-v2": PaperResult(39.0, 9.6),
    "halfcheetah-medium-expert-v2": PaperResult(63.3, 38.0),
    "hopper-medium-expert-v2": PaperResult(23.7, 6.0),
    "walker2d-medium-expert-v2": PaperResult(44.6, 12.9),
}


@dataclass(frozen=True)
class Config:
    result_directory_path: Path = RESULT_DIRECTORY_PATH


def write_result(
    *, task: str, seed: int, score: float, directory_path: Path = RESULT_DIRECTORY_PATH
) -> Path:
    file_path = directory_path / f"{task}-seed{seed}.json"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w") as file:
        json.dump(obj={"task": task, "seed": seed, "score": score}, fp=file, indent=2)
    return file_path


def main() -> None:
    config = tyro.cli(f=Config)
    logger.info(format_summary(result_directory_path=config.result_directory_path))


def format_summary(*, result_directory_path: Path = RESULT_DIRECTORY_PATH) -> str:
    scores = _read_scores(directory_path=result_directory_path)

    lines: list[str] = [
        _ROW_FORMAT.format("task", "ours", "theirs", "paper", "gap", "verdict", "seeds")
    ]
    for name, task in ALL_TASKS.items():
        our_scores = scores.get(name)
        if (
            our_scores is None
            or task.published_score is None
            or task.published_score_std is None
        ):
            continue
        mean = float(np.mean(our_scores))
        gap = mean - task.published_score
        paper = PAPER_RESULTS.get(name)
        lines.append(
            _ROW_FORMAT.format(
                name,
                f"{mean:.1f} +- {np.std(our_scores):.1f}",
                f"{task.published_score:.1f} +- {task.published_score_std:.1f}",
                "-"
                if paper is None
                else f"{paper.mean:.1f} +- {paper.standard_deviation:.1f}",
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
