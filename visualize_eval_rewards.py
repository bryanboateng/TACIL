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

import json
import re
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tyro
import wandb
from matplotlib.colors import LinearSegmentedColormap

from scripts.shared.paths import OUTPUT_DIRECTORY_PATH

mpl.use("Agg")

# Extract lip/noise/error from an eval note. Uses search (not a prefix match) so
# it also handles a variant prefix like "eval-sub0.25-lip-...".
NOTE_PATTERN = re.compile(r"lip-([^-]+)-action-([^-]+)-error-([^-]+)")

# Baseline runs carry no lip/action/error at all; "gpi" separates the two kinds.
BASELINE_MARKER = "noaug"
GEOMETRIC_MARKER = "gpi"

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

CACHE_PATH = OUTPUT_DIRECTORY_PATH / "eval_cache.json"
KEY_SEPARATOR = "|"

# Where the diverging scale ends, as a percentile of |delta| over all projects.
DELTA_PERCENTILE = 95

# Summary keys of the augmentation run that the metrics below are built from.
SYNTHETIC_TRANSITIONS = "4_augmented_episodes/synthetic_transition_count"
COPIED_EXPERT_TRANSITIONS = "4_augmented_episodes/copied_expert_transition_count"
COMBINED_TRANSITIONS = "5_combined_dataset/transition_count"
SYNTHETIC_PART_LENGTH_MEAN = "4_augmented_episodes/synthetic_part_length_mean"

AUGMENTATION_SUMMARY_KEYS = [
    SYNTHETIC_TRANSITIONS,
    COPIED_EXPERT_TRANSITIONS,
    COMBINED_TRANSITIONS,
    SYNTHETIC_PART_LENGTH_MEAN,
]


def _synthetic_per_source_transition(summary: dict[str, float]) -> float | None:
    """Synthetic transitions per transition of the original dataset.

    Deliberately not the logged synthetic_transition_fraction: that divides by the
    combined dataset, which every augmented episode inflates with 15 copied expert
    steps. Those copies are not new information, so they do not belong in the
    denominator of "how much was added".
    """
    source = (
        summary.get(COMBINED_TRANSITIONS, 0)
        - summary.get(SYNTHETIC_TRANSITIONS, 0)
        - summary.get(COPIED_EXPERT_TRANSITIONS, 0)
    )
    if source <= 0 or SYNTHETIC_TRANSITIONS not in summary:
        return None
    return summary[SYNTHETIC_TRANSITIONS] / source


# Properties of the augmented dataset itself: row label, number format, value.
AUGMENTATION_METRICS = [
    (
        "synthetic transitions\nper original one",
        "{:.2f}",
        _synthetic_per_source_transition,
    ),
    (
        "mean synthetic\npart length",
        "{:.1f}",
        lambda summary: summary.get(SYNTHETIC_PART_LENGTH_MEAN),
    ),
]


@dataclass(frozen=True)
class ProjectData:
    """Everything one project contributes to the figures."""

    grid: dict[tuple[str, str, str], dict[str, float]]
    diffusion_baseline: dict[str, float]
    geometric_baseline: dict[str, float]
    # (lip, noise, error) -> {summary key: value} of the augmentation run.
    augmentation: dict[tuple[str, str, str], dict[str, float]]


def main(
    projects: list[str],
    scale: Literal["condition", "global"],
    use_cache: bool = False,
) -> None:
    """Render the figures from the current state of the W&B runs.

    Pulling ~200 runs takes minutes and dominates the runtime, while rendering
    takes seconds. Pass --use-cache while iterating on the design to reuse the
    numbers of the previous fetch. It is off by default so a figure is never
    silently built from stale data.
    """
    rewards_per_project = _projects(projects=projects, use_cache=use_cache)
    for project, rewards in rewards_per_project.items():
        print(f"\n=== {project} ===")
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
        absolute_path = OUTPUT_DIRECTORY_PATH / f"eval_reward_absolute_{project}.png"
        _plot(rewards=rewards, project=project, output_path=absolute_path, scale=scale)
        print(f"Wrote {absolute_path}")

        augmentation_path = OUTPUT_DIRECTORY_PATH / f"augmentation_{project}.png"
        _plot_augmentation(
            rewards=rewards, project=project, output_path=augmentation_path
        )
        print(f"Wrote {augmentation_path}")

        if not rewards.diffusion_baseline:
            print(f"No '{BASELINE_MARKER}' baseline in {project}; skipping delta.")
            continue
        delta_path = OUTPUT_DIRECTORY_PATH / f"eval_reward_delta_{project}.png"
        _plot(
            rewards=rewards,
            project=project,
            output_path=delta_path,
            scale=scale,
            delta_limit=delta_limit,
        )
        print(f"Wrote {delta_path}")


def _projects(*, projects: list[str], use_cache: bool) -> dict[str, ProjectData]:
    cache = _read_cache() if use_cache else {}
    reused = [project for project in projects if project in cache]
    missing = [project for project in projects if project not in cache]

    if reused:
        _warn_about_cache(projects=reused)

    if missing:
        api = wandb.Api()
        entity = api.default_entity
        for project in missing:
            print(f"Fetching {entity}/{project} from W&B ...")
            cache[project] = _collect(api=api, run_path=f"{entity}/{project}")
        _write_cache(cache=cache)
        print(f"Cached {len(cache)} projects in {CACHE_PATH}")

    return {project: cache[project] for project in projects}


def _warn_about_cache(*, projects: list[str]) -> None:
    timestamp = datetime.fromtimestamp(CACHE_PATH.stat().st_mtime)
    lines = [
        "READING CACHED DATA - THE FIGURES MAY BE OUT OF DATE",
        f"cache written {timestamp:%Y-%m-%d %H:%M}",
        f"from cache: {', '.join(projects)}",
        "re-run without --use-cache to fetch the current runs",
    ]
    width = max(len(line) for line in lines) + 4
    print("\n" + "!" * width)
    for line in lines:
        print(f"! {line.ljust(width - 4)} !")
    print("!" * width + "\n")


def _read_cache() -> dict[str, ProjectData]:
    if not CACHE_PATH.exists():
        return {}
    document = json.loads(CACHE_PATH.read_text())
    return {
        project: ProjectData(
            grid={_key_from_text(text): v for text, v in entry["grid"].items()},
            diffusion_baseline=entry["diffusion_baseline"],
            geometric_baseline=entry["geometric_baseline"],
            augmentation={
                _key_from_text(text): v for text, v in entry["augmentation"].items()
            },
        )
        for project, entry in document.items()
    }


def _write_cache(*, cache: dict[str, ProjectData]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(
            {
                project: {
                    "grid": {_key_to_text(k): v for k, v in data.grid.items()},
                    "diffusion_baseline": data.diffusion_baseline,
                    "geometric_baseline": data.geometric_baseline,
                    "augmentation": {
                        _key_to_text(k): v for k, v in data.augmentation.items()
                    },
                }
                for project, data in cache.items()
            },
            indent=2,
        )
    )


# JSON has no tuple keys, so the (lip, noise, error) triple travels as one string.
def _key_to_text(key: tuple[str, str, str]) -> str:
    return KEY_SEPARATOR.join(key)


def _key_from_text(text: str) -> tuple[str, str, str]:
    lip, noise, error = text.split(KEY_SEPARATOR)
    return lip, noise, error


def _collect(*, api: wandb.Api, run_path: str) -> ProjectData:
    # (lip, noise, error) -> {condition: reward}, keeping the most complete run.
    best: dict[tuple[str, str, str], dict] = {}
    diffusion_baseline: dict[str, float] = {}
    geometric_baseline: dict[str, float] = {}
    augmentation: dict[tuple[str, str, str], dict[str, float]] = {}

    for run in api.runs(path=run_path):
        if run.job_type == "create-tacil-dataset":
            match = NOTE_PATTERN.search(run.notes or "")
            if match is not None:
                augmentation[match.groups()] = {
                    key: float(run.summary[key])
                    for key in AUGMENTATION_SUMMARY_KEYS
                    if isinstance(run.summary.get(key), (int, float))
                }
            continue
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

    return ProjectData(
        grid={key: value["rewards"] for key, value in best.items()},
        diffusion_baseline=diffusion_baseline,
        geometric_baseline=geometric_baseline,
        augmentation=augmentation,
    )


def _condition_rewards(*, run) -> dict[str, float]:
    condition_rewards: dict[str, float] = {}
    for summary_key, value in run.summary.items():
        summary_key = str(summary_key)
        if not summary_key.endswith("/reward_mean"):
            continue
        condition = summary_key.rsplit("/", maxsplit=1)[0]
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


def _report(*, rewards: ProjectData) -> None:
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


def _plot_augmentation(*, rewards: ProjectData, project: str, output_path: Path) -> None:
    """How much synthetic data each combination actually produced.

    A property of the dataset rather than of an eval condition, so this is its own
    figure: one row per metric, each on its own scale because the two have
    different units.
    """
    lips, noises, errors = _axes_values(grid=rewards.grid)
    colormap = plt.get_cmap(SEQUENTIAL_COLORMAP).with_extremes(bad=MISSING_COLOR)

    figure, axes = plt.subplots(
        nrows=len(AUGMENTATION_METRICS),
        ncols=len(lips),
        figsize=(2.4 * len(lips) + 2.4, 1.9 * len(AUGMENTATION_METRICS) + 1.4),
        squeeze=False,
    )

    for row, (label, number_format, metric) in enumerate(AUGMENTATION_METRICS):
        values = [
            value
            for summary in rewards.augmentation.values()
            for value in [metric(summary)]
            if value is not None
        ]
        vmin, vmax = (min(values), max(values)) if values else (0.0, 1.0)

        mesh = None
        for column, lip in enumerate(lips):
            ax = axes[row][column]
            grid = np.full((len(noises), len(errors)), np.nan)
            for i, noise in enumerate(noises):
                for j, error in enumerate(errors):
                    summary = rewards.augmentation.get((lip, noise, error))
                    value = None if summary is None else metric(summary)
                    if value is not None:
                        grid[i, j] = value

            mesh = ax.imshow(
                np.ma.masked_invalid(grid),
                cmap=colormap,
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
                origin="lower",
            )
            span = (vmax - vmin) or 1.0
            for i in range(len(noises)):
                for j in range(len(errors)):
                    if np.isnan(grid[i, j]):
                        continue
                    ax.text(
                        j,
                        i,
                        number_format.format(grid[i, j]),
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white" if grid[i, j] > vmin + 0.62 * span else "#0b0b0b",
                    )

            ax.set_xticks(range(len(errors)))
            ax.set_yticks(range(len(noises)))
            ax.set_xticklabels(
                errors if row == len(AUGMENTATION_METRICS) - 1 else [], fontsize=7
            )
            ax.set_yticklabels(noises if column == 0 else [], fontsize=7)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(f"lip = {lip}", fontsize=9)
            if column == 0:
                ax.set_ylabel(label, fontsize=8)
            if row == len(AUGMENTATION_METRICS) - 1:
                ax.set_xlabel("error_limit", fontsize=7)

        _add_row_colorbar(figure=figure, row_axes=list(axes[row]), mesh=mesh)

    figure.suptitle(
        f"{project}: what the augmentation produced\n"
        "rows: dataset property — y within tile: action_noise",
        fontsize=11,
    )
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig=figure)


def _plot(
    *,
    rewards: ProjectData,
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

    # A narrow leading column carries the baselines on the same colour scale as
    # the tiles, so GPI can be read against the augmentation grid instead of
    # sitting outside it as text.
    figure, axes = plt.subplots(
        nrows=len(CONDITIONS),
        ncols=len(lips) + 1,
        figsize=(2.4 * len(lips) + 3.0, 1.9 * len(CONDITIONS) + 1.2),
        width_ratios=[0.42] + [1.0] * len(lips),
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

        _plot_reference(
            ax=axes[row][0],
            rewards=rewards,
            condition=condition,
            colormap=colormap,
            vmin=vmin,
            vmax=vmax,
            is_delta=is_delta,
            is_top_row=row == 0,
        )

        for offset, lip in enumerate(lips):
            column = offset + 1
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
            ax.set_yticklabels(noises if column == 1 else [], fontsize=7)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(f"lip = {lip}", fontsize=9)
            if row == len(CONDITIONS) - 1:
                ax.set_xlabel("error_limit", fontsize=7)

        axes[row][0].set_ylabel(condition, fontsize=8)

        if not is_delta and scale == "condition":
            # Each row has its own scale, so each row gets its own key. Without
            # one, equal colours in neighbouring rows read as equal rewards.
            _add_row_colorbar(figure=figure, row_axes=list(axes[row]), mesh=mesh)

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


def _plot_reference(
    *,
    ax,
    rewards: ProjectData,
    condition: str,
    colormap,
    vmin: float,
    vmax: float,
    is_delta: bool,
    is_top_row: bool,
) -> None:
    diffusion = rewards.diffusion_baseline.get(condition)
    geometric = rewards.geometric_baseline.get(condition)

    if is_delta:
        # dp is the reference and would be a constant zero, so only gpi is shown:
        # how the geometric policy does against the same no-augmentation policy.
        labels = ["gpi"]
        values = [
            geometric - diffusion
            if geometric is not None and diffusion is not None
            else np.nan
        ]
    else:
        labels = ["gpi", "dp"]
        values = [
            geometric if geometric is not None else np.nan,
            diffusion if diffusion is not None else np.nan,
        ]

    column = np.array(values).reshape(-1, 1)
    ax.imshow(
        np.ma.masked_invalid(column),
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        origin="lower",
    )
    _annotate(ax=ax, grid=column, vmin=vmin, vmax=vmax, is_delta=is_delta)

    ax.set_xticks([])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if is_top_row:
        ax.set_title("no aug.", fontsize=9)


def _add_row_colorbar(*, figure, row_axes: list, mesh) -> None:
    colorbar = figure.colorbar(mesh, ax=row_axes, fraction=0.015, pad=0.012)
    colorbar.ax.tick_params(labelsize=6, length=0)
    colorbar.outline.set_visible(False)


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
    tyro.cli(f=main)
