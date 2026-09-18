"""Writes the CCIL replication table as LaTeX, ready to \\input into the thesis.

uv run python -m scripts.data.augmentation.ccil.replication.latex \\
    --ccil-output-directory-path=~/masters-thesis/CCIL/output

That writes ccil_replication.tex next to the reward files, which needs
\\usepackage{booktabs}. To look at it, compile ccil_replication_preview.tex.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from scripts.data.augmentation.ccil.replication.results import (
    PAPER_RESULTS,
    VARIANT_LABELS,
)
from scripts.shared.log import create_logger
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH

logger = create_logger()

RESULT_DIRECTORY_PATH = OUTPUT_DIRECTORY_PATH / "replication" / "ccil"

SOURCE_LABELS = ("Ours", "Ref.", "Paper")


@dataclass(frozen=True)
class Config:
    ccil_output_directory_path: Path | None = None
    result_directory_path: Path = RESULT_DIRECTORY_PATH
    output_file_path: Path = RESULT_DIRECTORY_PATH / "ccil_replication.tex"


PREVIEW_DOCUMENT = r"""\documentclass[a4paper,11pt]{article}
\usepackage[margin=18mm]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{amsmath}
\pagestyle{empty}
\begin{document}
\input{ccil_replication.tex}
\end{document}
"""


def main() -> None:
    config = tyro.cli(f=Config)

    ours = _read_seed_means(directory_path=config.result_directory_path)
    if not ours:
        raise FileNotFoundError(
            f"No reward files under {config.result_directory_path}. "
            "Run scripts.data.augmentation.ccil.replication.evaluate first."
        )
    logger.info("Read %d task and variant combinations of our own.", len(ours))

    theirs = (
        {}
        if config.ccil_output_directory_path is None
        else _read_their_seed_means(
            directory_path=config.ccil_output_directory_path.expanduser()
        )
    )
    logger.info("Read %d task and variant combinations of theirs.", len(theirs))

    config.output_file_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_file_path.write_text(_render_table(ours=ours, theirs=theirs))
    _write_preview(output_file_path=config.output_file_path)
    logger.info("Wrote %s.", config.output_file_path)


# ---- Private --------------------------------------------


def _write_preview(*, output_file_path: Path) -> None:
    """A standalone wrapper, so the table can be compiled on its own."""
    preview_file_path = output_file_path.with_name(
        output_file_path.stem + "_preview.tex"
    )
    preview_file_path.write_text(PREVIEW_DOCUMENT)
    logger.info("Wrote %s.", preview_file_path)


def _read_seed_means(*, directory_path: Path) -> dict[tuple[str, str], list[float]]:
    seed_means: dict[tuple[str, str], list[float]] = {}
    for rewards_file_path in sorted(directory_path.glob("*/*.pkl")):
        task = rewards_file_path.parent.name
        variant = rewards_file_path.stem.split("-seed")[0]
        with rewards_file_path.open("rb") as file:
            rewards = pickle.load(file)
        seed_means.setdefault((task, variant), []).append(float(np.mean(rewards)))
    return seed_means


def _read_their_seed_means(
    *, directory_path: Path
) -> dict[tuple[str, str], list[float]]:
    seed_means: dict[tuple[str, str], list[float]] = {}
    for rewards_file_path in sorted(
        directory_path.glob("*/seed*/*/policy/*/rewards_noise_*.pkl")
    ):
        parts = rewards_file_path.parts
        task = parts[-6]
        policy_directory_name = parts[-2]
        if policy_directory_name == "naive":
            variant = "none"
        elif policy_directory_name.startswith("noise_"):
            variant = "noise"
        else:
            variant = "ccil"
        with rewards_file_path.open("rb") as file:
            rewards = pickle.load(file)
        seed_means.setdefault((task, variant), []).append(float(np.mean(rewards)))
    return seed_means


def _render_table(
    *,
    ours: dict[tuple[str, str], list[float]],
    theirs: dict[tuple[str, str], list[float]],
) -> str:
    tasks = sorted({task for task, _variant in ours})
    rows = "\n".join(
        _render_row(task=task, ours=ours, theirs=theirs) for task in tasks
    )
    seed_count = min(len(values) for values in ours.values())
    return rf"""\begin{{table}}[htbp]
  \centering
  \caption{{CCIL replication: episode return over {seed_count} seeds.}}
  \label{{tab:ccil-replication}}
  \footnotesize
  \setlength{{\tabcolsep}}{{3pt}}
  \begin{{tabular}}{{l{"@{\\hspace{7pt}}".join(["rrr"] * len(VARIANT_LABELS))}}}
    \toprule
{_render_head()}
    \midrule
{rows}
    \bottomrule
  \end{{tabular}}
\end{{table}}
"""


def _render_head() -> str:
    group_cells = []
    rules = []
    first = 2
    for _variant, label in VARIANT_LABELS:
        last = first + len(SOURCE_LABELS) - 1
        group_cells.append(rf"\multicolumn{{{len(SOURCE_LABELS)}}}{{c}}{{{label}}}")
        rules.append(rf"\cmidrule(lr){{{first}-{last}}}")
        first = last + 1
    labels = " & ".join(SOURCE_LABELS * len(VARIANT_LABELS))
    return (
        f"    & {' & '.join(group_cells)} \\\\\n"
        f"    {' '.join(rules)}\n"
        f"    Task & {labels} \\\\"
    )


def _render_row(
    *,
    task: str,
    ours: dict[tuple[str, str], list[float]],
    theirs: dict[tuple[str, str], list[float]],
) -> str:
    cells = []
    for variant, _label in VARIANT_LABELS:
        paper = PAPER_RESULTS.get(task, {}).get(variant)
        cells.append(_format(values=ours.get((task, variant))))
        cells.append(_format(values=theirs.get((task, variant))))
        cells.append(
            "--"
            if paper is None
            else (
                rf"${paper.mean:.0f}$"
                rf"\,\scalebox{{0.8}}{{$\pm{paper.standard_deviation:.0f}$}}"
            )
        )
    return rf"    {task.replace('_', r'\_')} & {' & '.join(cells)} \\"


def _format(*, values: list[float] | None) -> str:
    if not values:
        return "--"
    if len(values) < 2:
        return rf"${np.mean(values):.0f}$"
    return (
        rf"${np.mean(values):.0f}$"
        rf"\,\scalebox{{0.8}}{{$\pm{np.std(a=values, ddof=1):.0f}$}}"
    )


if __name__ == "__main__":
    main()
