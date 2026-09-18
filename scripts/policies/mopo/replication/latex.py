"""Writes the MOPO replication table as LaTeX, ready to \\input into the thesis.

uv run python -m scripts.policies.mopo.replication.latex

That writes mopo_replication.tex next to the other replication output, which
needs \\usepackage{booktabs}. To look at it, compile mopo_replication_preview.tex.
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
import wandb

from scripts.policies.mopo.replication.results import PAPER_RESULTS, RESULT_DIRECTORY_PATH
from scripts.policies.mopo.replication.tasks import ALL_TASKS
from scripts.shared.log import create_logger

logger = create_logger()

SCORE_KEY = "4_evaluation/1_score_mean"


@dataclass(frozen=True)
class Config:
    project: str = "bryanboateng-team/mopo-replication"
    output_file_path: Path = RESULT_DIRECTORY_PATH / "mopo_replication.tex"


PREVIEW_DOCUMENT = r"""\documentclass[a4paper,11pt]{article}
\usepackage[margin=18mm]{geometry}
\usepackage{booktabs}
\usepackage{amsmath}
\pagestyle{empty}
\begin{document}
\input{mopo_replication.tex}
\end{document}
"""


def main() -> None:
    config = tyro.cli(f=Config)

    scores = _read_scores(project=config.project)
    config.output_file_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_file_path.write_text(_render_table(scores=scores))
    _write_preview(output_file_path=config.output_file_path)
    logger.info("Wrote %s (%d tasks).", config.output_file_path, len(scores))


# ---- Private --------------------------------------------


def _write_preview(*, output_file_path: Path) -> None:
    """A standalone wrapper, so the table can be compiled on its own."""
    preview_file_path = output_file_path.with_name(
        output_file_path.stem + "_preview.tex"
    )
    preview_file_path.write_text(PREVIEW_DOCUMENT)
    logger.info("Wrote %s.", preview_file_path)


def _read_scores(*, project: str) -> dict[str, dict[int, float]]:
    scores: dict[str, dict[int, float]] = defaultdict(dict)
    for run in wandb.Api().runs(project, per_page=400):
        score = run.summary.get(SCORE_KEY)
        if run.state == "finished" and score is not None:
            scores[run.config["task"]][run.config["seed"]] = score
    return scores


def _render_table(*, scores: dict[str, dict[int, float]]) -> str:
    rows = "\n".join(
        _render_row(name=name, seed_scores=scores[name])
        for name in ALL_TASKS
        if name in scores
    )
    seed_count = min(len(seed_scores) for seed_scores in scores.values())
    return rf"""\begin{{table}}[htbp]
  \centering
  \caption{{MOPO on D4RL: normalised score $\pm$ its standard deviation across
  {seed_count} seeds, next to the score the reference implementation reports and
  the one the original paper reports.}}
  \label{{tab:mopo-replication}}
  \small
  \begin{{tabular}}{{lrrr}}
    \toprule
    Task & Ours & Reference & Paper \\
    \midrule
{rows}
    \bottomrule
  \end{{tabular}}
\end{{table}}
"""


def _render_row(*, name: str, seed_scores: dict[int, float]) -> str:
    task = ALL_TASKS[name]
    values = [seed_scores[seed] for seed in sorted(seed_scores)]
    paper = PAPER_RESULTS.get(name)
    return (
        rf"    {name.replace('_', '-')} "
        rf"& ${np.mean(values):.1f} \pm {np.std(a=values, ddof=1):.1f}$ "
        rf"& ${task.published_score:.1f} \pm {task.published_score_std:.1f}$ "
        + (
            r"& -- \\"
            if paper is None
            else rf"& ${paper.mean:.1f} \pm {paper.standard_deviation:.1f}$ \\"
        )
    )


if __name__ == "__main__":
    main()
