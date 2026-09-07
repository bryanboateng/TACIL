import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import tyro
import wandb

from scripts.evaluation.core.condition import ALL_CONDITIONS
from scripts.evaluation.evaluate_policy import ALL_CONDITIONS_LABEL, STAGE_NAME
from scripts.sweep.create_sweep import METHOD_PARAMETERS

CONDITION_LABELS = (ALL_CONDITIONS_LABEL, *(c.label for c in ALL_CONDITIONS))

# Keyed by a plain string, because a method read back off a run is one. A
# method that was never swept, such as an unaugmented baseline, has no entry
# and so no hyperparameters to group its seeds by.
SWEPT_PARAMETER_NAMES: dict[str, tuple[str, ...]] = {
    method: tuple(parameters) for method, parameters in METHOD_PARAMETERS.items()
}

_DATASET_PERCENTAGE_PATTERN = re.compile(r"dataset-(\d+)")


@dataclass(frozen=True)
class Config:
    project: str
    # What picks the winning config per method.
    selection_condition: str = ALL_CONDITIONS_LABEL


@dataclass(frozen=True)
class Group:
    """The runs that differ only in their seed."""

    dataset_percentage: int
    method: str
    # The swept hyperparameters as (name, value) pairs, empty for a method
    # without any. Pairs rather than a dict so the group can be a mapping key.
    parameters: tuple[tuple[str, Any], ...]


def main() -> None:
    config = tyro.cli(Config)
    api = wandb.Api()

    rewards_per_group: dict[Group, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for run in api.runs(config.project, per_page=400):
        if run.state != "finished":
            continue
        group = _read_group(run=run)
        if group is None:
            continue
        for label in CONDITION_LABELS:
            reward = run.summary.get(f"{STAGE_NAME}/{label}/1_reward_mean")
            if reward is not None:
                rewards_per_group[group][label].append(reward)

    best = _select_best_per_method(
        rewards_per_group=rewards_per_group,
        selection_condition=config.selection_condition,
    )

    print(f"Best config per method (selected by {config.selection_condition}):\n")
    for key in sorted(best):
        group = best[key]
        seed_count = len(rewards_per_group[group][config.selection_condition])
        print(
            f"### {group.dataset_percentage} {group.method}   "
            f"(best config, {seed_count} seeds)"
        )
        print("    config:", dict(group.parameters))
        for label in CONDITION_LABELS:
            summary = _summarize(values=rewards_per_group[group].get(label, []))
            if summary is None:
                continue
            mean, standard_error, count = summary
            marker = "  <-- selection" if label == config.selection_condition else ""
            print(
                f"    {label:28} {mean:.3f} ± {standard_error:.3f}  (n={count}){marker}"
            )
        print()


def _read_group(*, run: Any) -> Group | None:
    """Identify a run from what it logged itself.

    Nothing tags a run, so the method is read back off the policy that ran and
    the augmentation config it carries, and the dataset size off the artifact
    name. Both are written by the run itself, so a run cannot go missing from
    the summary because a tag was forgotten.
    """
    method = _read_method(run=run)
    if method is None:
        return None

    match = _DATASET_PERCENTAGE_PATTERN.search(
        run.config.get("expert_dataset_artifact", "")
    )
    if match is None:
        return None

    return Group(
        dataset_percentage=int(match.group(1)),
        method=method,
        parameters=tuple(
            (name, _read_parameter(config=run.config, name=name))
            for name in SWEPT_PARAMETER_NAMES.get(method, ())
        ),
    )


def _read_method(*, run: Any) -> str | None:
    """The method label, in the spelling the results page uses."""
    policy = run.job_type
    if policy == "gpi":
        return "gpi"
    if policy not in ("bc", "dp"):
        return None

    augmentation = run.config.get("augmentation")
    if augmentation is None:
        # A baseline: the policy trained on the expert data alone.
        return policy
    # The two augmentations have no type tag in the logged config, so they are
    # told apart by the section only CCIL has.
    return "ccil" if "labels" in augmentation else f"tacil-{policy}"


def _read_parameter(*, config: dict[str, Any], name: str) -> Any:
    """Read a dotted name, whether wandb stored it flat or nested."""
    if name in config:
        return _round(value=config[name])
    current: Any = config
    for part in name.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return _round(value=current)


def _round(*, value: Any) -> Any:
    """Floats that differ only in their last digits are the same config."""
    return round(value, 6) if isinstance(value, float) else value


def _select_best_per_method(
    *, rewards_per_group: dict[Group, dict[str, list[float]]], selection_condition: str
) -> dict[tuple[int, str], Group]:
    best: dict[tuple[int, str], Group] = {}
    best_mean: dict[tuple[int, str], float] = {}
    for group, rewards_per_condition in rewards_per_group.items():
        summary = _summarize(values=rewards_per_condition.get(selection_condition, []))
        if summary is None:
            continue
        mean = summary[0]
        key = (group.dataset_percentage, group.method)
        if key not in best_mean or mean > best_mean[key]:
            best[key] = group
            best_mean[key] = mean
    return best


def _summarize(*, values: list[float]) -> tuple[float, float, int] | None:
    """The mean, the standard error of that mean, and how many seeds it is over."""
    count = len(values)
    if count == 0:
        return None
    mean = sum(values) / count
    if count < 2:
        return mean, 0.0, count
    variance = sum((value - mean) ** 2 for value in values) / (count - 1)
    return mean, math.sqrt(variance) / math.sqrt(count), count


if __name__ == "__main__":
    main()
