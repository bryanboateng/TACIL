import math
from collections import defaultdict
from dataclasses import dataclass

import tyro

from scripts.evaluation.core.condition import ALL_CONDITIONS, VALIDATION_CONDITION
from scripts.evaluation.evaluate_policy import STAGE_NAME
from scripts.sweep.create_sweep import METHOD_PARAMETERS

import wandb


@dataclass(frozen=True)
class Config:
    project: str = "2026-08-24-final"
    entity: str = "bryanboateng-team"
    # The condition whose mean picks the winning config per method.
    selection_condition: str = VALIDATION_CONDITION.label


def _get_param(config: dict, dotted: str):
    if dotted in config:
        return config[dotted]
    current = config
    for part in dotted.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _mean_se(values: list[float]) -> tuple[float | None, float, int]:
    n = len(values)
    if n == 0:
        return None, 0.0, 0
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0, n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    return mean, math.sqrt(variance) / math.sqrt(n), n


def main() -> None:
    config = tyro.cli(Config)
    api = wandb.Api()

    # groups[(pct, method, config_key)][condition_label] = per-seed rewards
    groups: dict = defaultdict(lambda: defaultdict(list))
    group_config: dict = {}

    for run in api.runs(f"{config.entity}/{config.project}", per_page=400):
        if run.state != "finished":
            continue
        method = next((t for t in run.tags if "pct" not in t), None)
        pct = next((t for t in run.tags if "pct" in t), None)
        if method in METHOD_PARAMETERS:
            config_key = tuple(
                (key, round(value, 6) if isinstance(value, float) else value)
                for key in METHOD_PARAMETERS[method]
                for value in [_get_param(run.config, key)]
            )
        elif method is not None and method.endswith("-noaug"):
            # No augmentation means no swept hyperparameters, so all seeds of a
            # baseline are one group.
            config_key = ()
        else:
            continue
        group = (pct, method, config_key)
        group_config[group] = dict(config_key)
        for condition in ALL_CONDITIONS:
            reward = run.summary.get(f"{STAGE_NAME}/{condition.label}/reward_mean")
            if reward is not None:
                groups[group][condition.label].append(reward)

    # Best config per (pct, method), chosen by the selection condition's mean.
    best: dict = {}
    for group, per_condition in groups.items():
        pct, method, _ = group
        mean, _, _ = _mean_se(per_condition.get(config.selection_condition, []))
        if mean is None:
            continue
        key = (pct, method)
        if key not in best or mean > best[key][1]:
            best[key] = (group, mean)

    print(f"Best config per method (selected by {config.selection_condition}):\n")
    for key in sorted(best):
        group, _ = best[key]
        pct, method = key
        n_seeds = len(next(iter(groups[group].values())))
        print(f"### {pct} {method}   (best config, {n_seeds} seeds)")
        print("    config:", group_config[group])
        for condition in ALL_CONDITIONS:
            mean, standard_error, n = _mean_se(groups[group].get(condition.label, []))
            if mean is not None:
                marker = (
                    "  <-- selection"
                    if condition.label == config.selection_condition
                    else ""
                )
                print(
                    f"    {condition.label:26} {mean:.3f} ± {standard_error:.3f}  (n={n}){marker}"
                )
        print()


if __name__ == "__main__":
    main()
