import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from scripts.shared.log import create_logger
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH

logger = create_logger()

VARIANT_LABELS = (("none", "BC"), ("noise", "NoiseBC"), ("ccil", "CCIL"))


@dataclass(frozen=True)
class PaperResult:
    mean: float
    standard_deviation: float


PAPER_RESULTS: dict[str, dict[str, PaperResult]] = {
    "hopper": {
        "none": PaperResult(1983.98, 672.66),
        "noise": PaperResult(1563.56, 1012.02),
        "ccil": PaperResult(2631.25, 303.86),
    },
    "walker2d": {
        "none": PaperResult(1922.55, 1410.09),
        "noise": PaperResult(2893.21, 1076.89),
        "ccil": PaperResult(3538.48, 573.23),
    },
    "ant": {
        "none": PaperResult(2965.20, 202.71),
        "noise": PaperResult(3776.65, 442.13),
        "ccil": PaperResult(3338.35, 474.17),
    },
    "halfcheetah": {
        "none": PaperResult(8309.31, 795.30),
        "noise": PaperResult(8468.98, 738.83),
        "ccil": PaperResult(8757.38, 379.12),
    },
    "coffee_pull": {
        "none": PaperResult(3552.59, 233.41),
        "noise": PaperResult(3072.86, 785.91),
        "ccil": PaperResult(4168.46, 192.98),
    },
    "button": {
        "none": PaperResult(3693.02, 104.99),
        "noise": PaperResult(3663.44, 63.10),
        "ccil": PaperResult(3775.22, 91.24),
    },
    "coffee_push": {
        "none": PaperResult(1288.19, 746.37),
        "noise": PaperResult(2551.11, 857.79),
        "ccil": PaperResult(2484.19, 976.03),
    },
    "drawer_close": {
        "none": PaperResult(3247.06, 468.73),
        "noise": PaperResult(4226.71, 18.90),
        "ccil": PaperResult(4145.45, 76.23),
    },
}

_ROW_FORMAT = "{:<14}{:<10}{:>24}{:>24}{:>7}"


@dataclass(frozen=True)
class Config:
    directory_path: Path = OUTPUT_DIRECTORY_PATH / "replication"


def main() -> None:
    config = tyro.cli(f=Config)
    logger.info(
        "Replication against the CCIL paper:\n%s",
        format_summary(directory_path=config.directory_path),
    )


def format_summary(*, directory_path: Path) -> str:
    seed_means = _read_seed_means(directory_path=directory_path)
    if not seed_means:
        return f"No scored policies under {directory_path}."

    lines: list[str] = [_ROW_FORMAT.format("task", "variant", "ours", "paper", "seeds")]
    for task in sorted({task for task, _ in seed_means}):
        for variant, label in VARIANT_LABELS:
            means = seed_means.get((task, variant))
            if means is None:
                continue
            paper = PAPER_RESULTS.get(task, {}).get(variant)
            lines.append(
                _ROW_FORMAT.format(
                    task,
                    label,
                    f"{np.mean(means):.2f} +- {np.std(means):.2f}",
                    "-"
                    if paper is None
                    else f"{paper.mean:.2f} +- {paper.standard_deviation:.2f}",
                    len(means),
                )
            )
    return "\n".join(lines)


# ---- Private --------------------------------------------


def _read_seed_means(*, directory_path: Path) -> dict[tuple[str, str], list[float]]:
    seed_means: dict[tuple[str, str], list[float]] = {}
    for rewards_file_path in sorted(directory_path.glob("*/*.pkl")):
        task = rewards_file_path.parent.name
        variant = rewards_file_path.stem.split("-seed")[0]
        with rewards_file_path.open("rb") as file:
            rewards = pickle.load(file)
        seed_means.setdefault((task, variant), []).append(float(np.mean(rewards)))
    return seed_means


if __name__ == "__main__":
    main()
