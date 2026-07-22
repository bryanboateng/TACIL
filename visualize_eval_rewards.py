"""Render the evaluation reward landscape from W&B into faceted heatmaps.

Two figures per project. The absolute one shows reward_mean itself; the delta one
shows reward_mean minus the no-augmentation diffusion-policy baseline, which is
the figure that actually answers "did the augmentation help". Rows are the six
eval conditions, columns the dynamics-model Lipschitz value, and each tile is a
noise x error_limit grid. Combinations that were never produced stay grey.

    uv run python visualize_eval_rewards.py --entity bryanboateng-team \
        --projects opq opq-sub025

The delta figures of all projects share one symmetric colour scale, so the 100 %
and 25 % figures can be laid side by side and compared directly.

Re-run any time as more eval runs finish; it always reflects the current state.
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import wandb
from matplotlib.colors import LinearSegmentedColormap

mpl.use("Agg")

# Extract lip/noise/error from an eval note. Uses search (not a prefix match) so
# it also handles a variant prefix like "eval-sub0.25-lip-...".
NOTE_PATTERN = re.compile(r"lip-([^-]+)-action-([^-]+)-error-([^-]+)")

# Baseline runs carry no lip/action/error at all; "gpi" separates the two kinds.
BASELINE_MARKER = "noaug"
GEOMETRIC_MARKER = "gpi"

# Early runs used the old condition labels; map them onto the current scheme.
LABEL_ALIASES = {
    "damping_grippy_disturbance_none": "grippy_undisturbed",
    "damping_grippy_disturbance_low": "grippy_low_disturbance",
    "damping_grippy_disturbance_high": "grippy_high_disturbance",
    "damping_slippery_disturbance_none": "slippery_undisturbed",
    "damping_slippery_disturbance_low": "slippery_low_disturbance",
    "damping_slippery_disturbance_high": "slippery_high_disturbance",
}

CONDITIONS = [
    "grippy_undisturbed",
    "grippy_low_disturbance",
    "grippy_high_disturbance",
    "slippery_undisturbed",
    "slippery_low_disturbance",
    "slippery_high_disturbance",
]

SEQUENTIAL_COLORMAP = "Blues"

# Diverging pair: blue for better, red for worse, with a neutral grey midpoint so
# "no change" reads as nothing rather than as a colour of its own.
DIVERGING_COLORMAP = LinearSegmentedColormap.from_list(
    "reward_delta", ["#e34948", "#f0efec", "#2a78d6"]
)

MISSING_COLOR = "#e4e4e1"

# Where the diverging scale ends, as a percentile of |delta| over all projects.
DELTA_PERCENTILE = 95


@dataclass(frozen=True)
class ProjectRewards:
    """Everything one project contributes to the figures."""

    grid: dict[tuple[str, str, str], dict[str, float]]
    diffusion_baseline: dict[str, float]
    geometric_baseline: dict[str, float]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projects", nargs="+", default=["opq"])
    parser.add_argument("--entity", default=None)
    parser.add_argument("--output-directory", default=".")
    parser.add_argument(
        "--scale",
        choices=["condition", "global"],
        default="condition",
        help="absolute figures only. condition: colour scaled per row (see "
        "within-condition differences); global: one scale across all rows.",
    )
    arguments = parser.parse_args()

    api = wandb.Api()
    entity = arguments.entity or api.default_entity
    output_directory = Path(arguments.output_directory)

    rewards_per_project = {
        project: _collect(api=api, run_path=f"{entity}/{project}")
        for project in arguments.projects
    }
    for project, rewards in rewards_per_project.items():
        print(f"\n=== {entity}/{project} ===")
        _report(rewards=rewards)

    # One symmetric limit across all projects, so their delta figures compare.
    deltas = [
        reward - rewards.diffusion_baseline[condition]
        for rewards in rewards_per_project.values()
        for conditions in rewards.grid.values()
        for condition, reward in conditions.items()
        if condition in rewards.diffusion_baseline
    ]
    # A robust limit rather than the maximum: a couple of large negative outliers
    # would otherwise wash the remaining tiles out to near-white and leave the
    # colour carrying no information at all. Tiles beyond the limit clip to the
    # end of the ramp; their printed number still gives the exact value.
    delta_limit = (
        float(np.percentile(np.abs(deltas), DELTA_PERCENTILE)) if deltas else 1.0
    ) or 1.0

    print()
    for project, rewards in rewards_per_project.items():
        absolute_path = output_directory / f"eval_reward_absolute_{project}.png"
        _plot(
            rewards=rewards,
            project=project,
            output_path=absolute_path,
            scale=arguments.scale,
        )
        print(f"Wrote {absolute_path}")

        if not rewards.diffusion_baseline:
            print(f"No '{BASELINE_MARKER}' baseline in {project}; skipping delta.")
            continue
        delta_path = output_directory / f"eval_reward_delta_{project}.png"
        _plot(
            rewards=rewards,
            project=project,
            output_path=delta_path,
            scale=arguments.scale,
            delta_limit=delta_limit,
        )
        print(f"Wrote {delta_path}")


def _collect(*, api: wandb.Api, run_path: str) -> ProjectRewards:
    # (lip, noise, error) -> {condition: reward}, keeping the most complete run.
    best: dict[tuple[str, str, str], dict] = {}
    diffusion_baseline: dict[str, float] = {}
    geometric_baseline: dict[str, float] = {}

    for run in api.runs(path=run_path):
        if run.job_type != "evaluate-policy":
            continue
        note = run.notes or ""
        condition_rewards = _condition_rewards(run=run)
        if not condition_rewards:
            continue

        match = NOTE_PATTERN.search(note)
        if match is None:
            if BASELINE_MARKER not in note:
                continue
            if GEOMETRIC_MARKER in note:
                geometric_baseline = condition_rewards
            else:
                diffusion_baseline = condition_rewards
            continue

        key = match.groups()  # (lip, noise, error)
        # Prefer the run covering the most conditions, then the newest.
        candidate = (len(condition_rewards), run.created_at)
        if key not in best or candidate > best[key]["rank"]:
            best[key] = {"rank": candidate, "rewards": condition_rewards}

    return ProjectRewards(
        grid={key: value["rewards"] for key, value in best.items()},
        diffusion_baseline=diffusion_baseline,
        geometric_baseline=geometric_baseline,
    )


def _condition_rewards(*, run) -> dict[str, float]:
    condition_rewards: dict[str, float] = {}
    for summary_key, value in run.summary.items():
        summary_key = str(summary_key)
        if not summary_key.endswith("/reward_mean"):
            continue
        condition = summary_key.rsplit("/", maxsplit=1)[0]
        condition = LABEL_ALIASES.get(condition, condition)
        if condition in CONDITIONS and isinstance(value, (int, float)):
            condition_rewards[condition] = float(value)
    return condition_rewards


def _axes_values(*, grid: dict) -> tuple[list[str], list[str], list[str]]:
    lips = sorted({key[0] for key in grid}, key=_lip_sort_key)
    noises = sorted({key[1] for key in grid}, key=float)
    errors = sorted({key[2] for key in grid}, key=float)
    return lips, noises, errors


def _lip_sort_key(lip: str) -> float:
    # null = no Lipschitz bound = effectively infinite, so it sorts after 15.
    return float("inf") if lip == "null" else float(lip)


def _report(*, rewards: ProjectRewards) -> None:
    lips, noises, errors = _axes_values(grid=rewards.grid)
    expected = len(lips) * len(noises) * len(errors)
    print(f"Combinations with data: {len(rewards.grid)} (grid would be {expected})")
    incomplete = {
        key: len(value)
        for key, value in rewards.grid.items()
        if len(value) < len(CONDITIONS)
    }
    if incomplete:
        print(f"Incomplete combinations (fewer than {len(CONDITIONS)} conditions):")
        for key, count in sorted(incomplete.items()):
            print(f"  lip={key[0]} noise={key[1]} error={key[2]}: {count} conditions")
    for name, baseline in (
        ("dp-noaug", rewards.diffusion_baseline),
        ("gpi-noaug", rewards.geometric_baseline),
    ):
        print(f"{name}: {'missing' if not baseline else f'{len(baseline)} conditions'}")


def _plot(
    *,
    rewards: ProjectRewards,
    project: str,
    output_path: Path,
    scale: str,
    delta_limit: float | None = None,
) -> None:
    is_delta = delta_limit is not None
    lips, noises, errors = _axes_values(grid=rewards.grid)

    colormap = DIVERGING_COLORMAP if is_delta else plt.get_cmap(SEQUENTIAL_COLORMAP)
    colormap = colormap.with_extremes(bad=MISSING_COLOR)

    finite = [r for conditions in rewards.grid.values() for r in conditions.values()]
    global_vmin, global_vmax = (min(finite), max(finite)) if finite else (0.0, 1.0)

    figure, axes = plt.subplots(
        nrows=len(CONDITIONS),
        ncols=len(lips),
        figsize=(2.4 * len(lips) + 2.0, 1.9 * len(CONDITIONS) + 1.2),
        squeeze=False,
    )

    mesh = None
    for row, condition in enumerate(CONDITIONS):
        reference = rewards.diffusion_baseline.get(condition)
        if is_delta:
            vmin, vmax = -delta_limit, delta_limit
        elif scale == "condition":
            row_values = [c[condition] for c in rewards.grid.values() if condition in c]
            vmin, vmax = (
                (min(row_values), max(row_values))
                if row_values
                else (global_vmin, global_vmax)
            )
        else:
            vmin, vmax = global_vmin, global_vmax

        for column, lip in enumerate(lips):
            ax = axes[row][column]
            grid = np.full((len(noises), len(errors)), np.nan)
            for i, noise in enumerate(noises):
                for j, error in enumerate(errors):
                    value = rewards.grid.get((lip, noise, error), {}).get(condition)
                    if value is None or (is_delta and reference is None):
                        continue
                    grid[i, j] = value - reference if is_delta else value

            mesh = ax.imshow(
                np.ma.masked_invalid(grid),
                cmap=colormap,
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
                origin="lower",
            )
            _annotate(ax=ax, grid=grid, vmin=vmin, vmax=vmax, is_delta=is_delta)

            ax.set_xticks(range(len(errors)))
            ax.set_yticks(range(len(noises)))
            ax.set_xticklabels(errors if row == len(CONDITIONS) - 1 else [], fontsize=7)
            ax.set_yticklabels(noises if column == 0 else [], fontsize=7)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(f"lip = {lip}", fontsize=9)
            if column == 0:
                ax.set_ylabel(
                    _row_label(
                        condition=condition,
                        rewards=rewards,
                        is_delta=is_delta,
                        vmin=vmin,
                        vmax=vmax,
                        scale=scale,
                    ),
                    fontsize=8,
                )
            if row == len(CONDITIONS) - 1:
                ax.set_xlabel("error_limit", fontsize=7)

    _finish(
        figure=figure,
        axes=axes,
        mesh=mesh,
        project=project,
        delta_limit=delta_limit,
        scale=scale,
    )
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig=figure)


def _annotate(*, ax, grid, vmin, vmax, is_delta: bool) -> None:
    # White ink only where the fill is dark enough to swallow black text: on the
    # diverging map that is both ends, on the sequential one only the top.
    span = max(abs(vmax), abs(vmin)) if is_delta else (vmax - vmin) or 1.0
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            value = grid[i, j]
            if np.isnan(value):
                continue
            if is_delta:
                is_dark = abs(value) > 0.62 * span
                text = f"{value:+.2f}"
            else:
                is_dark = value > vmin + 0.62 * span
                text = f"{value:.2f}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=6,
                color="white" if is_dark else "#0b0b0b",
            )


def _row_label(
    *, condition: str, rewards: ProjectRewards, is_delta: bool, vmin, vmax, scale: str
) -> str:
    label = condition
    if is_delta:
        # The baselines are what the tiles are measured against, so they belong on
        # the row itself rather than in a legend somewhere else.
        parts = []
        diffusion = rewards.diffusion_baseline.get(condition)
        geometric = rewards.geometric_baseline.get(condition)
        if diffusion is not None:
            parts.append(f"dp {diffusion:.2f}")
        if geometric is not None:
            parts.append(f"gpi {geometric:.2f}")
        if parts:
            label += "\n" + " · ".join(parts)
    elif scale == "condition":
        label += f"\n[{vmin:.2f}–{vmax:.2f}]"
    return label


def _finish(
    *, figure, axes, mesh, project: str, delta_limit: float | None, scale: str
) -> None:
    is_delta = delta_limit is not None
    if is_delta:
        title = (
            f"{project}: reward change vs. the no-augmentation policy\n"
            "blue = augmentation helps, red = it hurts  "
            "— rows: eval condition, y within tile: action_noise"
        )
    else:
        title = (
            f"{project}: policy reward across the augmentation space\n"
            "rows: eval condition — y within tile: action_noise"
        )
        if scale == "condition":
            title += "; colour scaled per row"
    figure.suptitle(title, fontsize=11)

    # A colourbar is only meaningful where one scale covers the whole figure.
    if mesh is not None and (is_delta or scale == "global"):
        colorbar = figure.colorbar(
            mesh,
            ax=axes,
            fraction=0.02,
            pad=0.01,
            label=(
                f"reward change (scale clipped at ±{delta_limit:.2f})"
                if is_delta
                else "reward_mean"
            ),
        )
        colorbar.outline.set_visible(False)


if __name__ == "__main__":
    main()
