"""Render the evaluation reward landscape from W&B into a faceted heatmap.

Rows = the six eval conditions, columns = the dynamics-model Lipschitz value
(null baseline first). Each tile is a noise x error_limit heatmap coloured by
reward_mean, on one shared colour scale. Missing combinations are left grey.

    uv run python visualize_eval_rewards.py --entity bryanboateng-team

Re-run any time as more eval runs finish; it always reflects the current state.
"""

import argparse
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import wandb

mpl.use("Agg")

NOTE_PATTERN = re.compile(r"eval-lip-(.+?)-action-(.+?)-error-(.+)")

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


def _lip_sort_key(lip: str) -> float:
    # null = no Lipschitz bound = effectively infinite, so it sorts after 15.
    return float("inf") if lip == "null" else float(lip)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="opq")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--output", default="eval_reward_landscape.png")
    parser.add_argument(
        "--scale",
        choices=["condition", "global"],
        default="condition",
        help="condition: colour scaled per row (see within-condition differences); "
        "global: one scale across all rows (compare conditions to each other).",
    )
    arguments = parser.parse_args()

    api = wandb.Api()
    entity = arguments.entity or api.default_entity

    rewards, lips, noises, errors = _collect(
        api=api, run_path=f"{entity}/{arguments.project}"
    )
    _report(rewards=rewards, lips=lips, noises=noises, errors=errors)
    _plot(
        rewards=rewards,
        lips=lips,
        noises=noises,
        errors=errors,
        output_path=arguments.output,
        scale=arguments.scale,
    )
    print(f"Wrote {arguments.output}")


def _collect(*, api: wandb.Api, run_path: str):
    # (lip, noise, error) -> {condition: reward}, keeping the most complete run.
    best: dict[tuple[str, str, str], dict] = {}
    for run in api.runs(path=run_path):
        if run.job_type != "evaluate-policy":
            continue
        match = NOTE_PATTERN.match(run.notes or "")
        if match is None:
            continue
        key = match.groups()  # (lip, noise, error)

        condition_rewards: dict[str, float] = {}
        for summary_key, value in run.summary.items():
            summary_key = str(summary_key)
            if not summary_key.endswith("/reward_mean"):
                continue
            condition = summary_key.rsplit("/", maxsplit=1)[0]
            condition = LABEL_ALIASES.get(condition, condition)
            if condition in CONDITIONS and isinstance(value, (int, float)):
                condition_rewards[condition] = float(value)

        if not condition_rewards:
            continue
        # Prefer the run covering the most conditions, then the newest.
        candidate = (len(condition_rewards), run.created_at)
        if key not in best or candidate > best[key]["rank"]:
            best[key] = {"rank": candidate, "rewards": condition_rewards}

    rewards = {key: value["rewards"] for key, value in best.items()}
    lips = sorted({key[0] for key in rewards}, key=_lip_sort_key)
    noises = sorted({key[1] for key in rewards}, key=float)
    errors = sorted({key[2] for key in rewards}, key=float)
    return rewards, lips, noises, errors


def _report(*, rewards, lips, noises, errors) -> None:
    expected = len(lips) * len(noises) * len(errors)
    print(f"Combinations with data: {len(rewards)} (grid would be {expected})")
    incomplete = {k: len(v) for k, v in rewards.items() if len(v) < len(CONDITIONS)}
    if incomplete:
        print(f"Incomplete combinations (fewer than {len(CONDITIONS)} conditions):")
        for key, count in sorted(incomplete.items()):
            print(f"  lip={key[0]} noise={key[1]} error={key[2]}: {count} conditions")


def _plot(*, rewards, lips, noises, errors, output_path, scale) -> None:
    finite = [r for cond in rewards.values() for r in cond.values()]
    global_vmin, global_vmax = (min(finite), max(finite)) if finite else (0.0, 1.0)

    fig, axes = plt.subplots(
        nrows=len(CONDITIONS),
        ncols=len(lips),
        figsize=(2.4 * len(lips) + 1.5, 1.9 * len(CONDITIONS) + 1.0),
        squeeze=False,
    )

    global_mesh = None
    for row, condition in enumerate(CONDITIONS):
        if scale == "condition":
            row_values = [c[condition] for c in rewards.values() if condition in c]
            vmin, vmax = (
                (min(row_values), max(row_values))
                if row_values
                else (global_vmin, global_vmax)
            )
        else:
            vmin, vmax = global_vmin, global_vmax
        midpoint = (vmin + vmax) / 2

        for column, lip in enumerate(lips):
            ax = axes[row][column]
            grid = np.full((len(noises), len(errors)), np.nan)
            for i, noise in enumerate(noises):
                for j, error in enumerate(errors):
                    grid[i, j] = rewards.get((lip, noise, error), {}).get(
                        condition, np.nan
                    )
            global_mesh = ax.imshow(
                grid, cmap="Blues", vmin=vmin, vmax=vmax, aspect="auto", origin="lower"
            )
            for i in range(len(noises)):
                for j in range(len(errors)):
                    if not np.isnan(grid[i, j]):
                        ax.text(
                            j,
                            i,
                            f"{grid[i, j]:.2f}",
                            ha="center",
                            va="center",
                            fontsize=6,
                            color="white" if grid[i, j] < midpoint else "black",
                        )

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
                label = condition
                if scale == "condition":
                    label += f"\n[{vmin:.2f}–{vmax:.2f}]"
                ax.set_ylabel(label, fontsize=8)
            if row == len(CONDITIONS) - 1:
                ax.set_xlabel("error_limit", fontsize=7)

    title = (
        "Policy reward across the augmentation space (rows: eval condition)  "
        "— y within tile: action_noise"
    )
    if scale == "condition":
        title += "\ncolour scaled per row (darker = higher within that condition)"
    fig.suptitle(title, fontsize=11)
    if scale == "global" and global_mesh is not None:
        colorbar = fig.colorbar(
            global_mesh, ax=axes, fraction=0.02, pad=0.01, label="reward_mean"
        )
        colorbar.outline.set_visible(False)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
