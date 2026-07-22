"""Report which combinations are missing at each stage of the pipeline.

Walks the lineage note-by-note (data- -> policy- -> eval-) and prints, per
project, what never made it to the next stage. Datasets are additionally checked
against the full grid implied by the values that do exist, so a combination that
was never created at all also shows up.

    uv run python report_gaps.py --entity bryanboateng-team opq opq-sub025

Missing policies and evals need no manual work: re-running generate_sweeps for
that phase schedules exactly them. Missing datasets have to be re-created by
hand, because that phase is not a sweep.
"""

import argparse
import itertools
import re

import wandb

from generate_sweeps import PHASES

# Matches the shared tail of every note, whatever prefix or tag precedes it.
NOTE_PATTERN = re.compile(r"lip-([^-]+)-action-([^-]+)-error-([^-]+)")

DATASET_JOB_TYPE = PHASES["dp"]["source_job_type"]
DATASET_ARTIFACT_TYPE = PHASES["dp"]["artifact_type"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("projects", nargs="+")
    parser.add_argument("--entity", default=None)
    arguments = parser.parse_args()

    api = wandb.Api()
    entity = arguments.entity or api.default_entity

    for project in arguments.projects:
        print(f"\n=== {entity}/{project} ===")
        _report_project(runs=list(api.runs(path=f"{entity}/{project}")))


def _report_project(*, runs: list) -> None:
    _report_missing_datasets(runs=runs)
    for phase_name in ("dp", "eval"):
        _report_missing_stage(runs=runs, phase=PHASES[phase_name])


def _report_missing_datasets(*, runs: list) -> None:
    # The grid is inferred from the values actually present, so this needs no
    # hard-coded axis definitions and adapts to whatever a project contains.
    # A run that logged no artifact produced no dataset, whatever its note says:
    # create_augmented_data logs its statistics before uploading, so a crash in
    # between leaves the run looking complete. generate_sweeps checks the same
    # thing, so both tools agree on what exists.
    combinations = {
        match.groups()
        for run in runs
        if run.job_type == DATASET_JOB_TYPE
        and any(a.type == DATASET_ARTIFACT_TYPE for a in run.logged_artifacts())
        for match in [NOTE_PATTERN.search(run.notes or "")]
        if match is not None
    }
    if not combinations:
        print("datasets: none found")
        return

    axes = [sorted({c[i] for c in combinations}) for i in range(3)]
    grid = set(itertools.product(*axes))
    missing = grid - combinations
    print(f"datasets: {len(combinations)}/{len(grid)} of the implied grid")
    for lip, noise, error in sorted(missing):
        print(f"  missing  lip={lip} action={noise} error={error}")


def _report_missing_stage(*, runs: list, phase: dict) -> None:
    # Only runs that actually emitted their artifact can feed the next stage;
    # this is the same condition generate_sweeps applies when building specs.
    upstream = {
        note
        for run in runs
        if run.job_type == phase["source_job_type"]
        and any(a.type == phase["artifact_type"] for a in run.logged_artifacts())
        for note in [(run.notes or "").strip()]
        if note.startswith(phase["note_prefix_from"])
    }
    expected = {
        phase["note_prefix_to"] + note[len(phase["note_prefix_from"]) :]
        for note in upstream
    }
    done = {
        note
        for run in runs
        if run.job_type == phase["consumer_job_type"]
        for note in [(run.notes or "").strip()]
        if note.startswith(phase["note_prefix_to"]) and phase["consumer_done"](run)
    }

    # Count only against what the grid expects. Baseline runs carry the same note
    # prefix but have no upstream combination, so counting every finished run
    # would report more than 100 % complete.
    missing = expected - done
    extra = done - expected
    label = phase["consumer_job_type"]
    print(f"{label}: {len(expected & done)}/{len(expected)} complete")
    for note in sorted(missing):
        print(f"  missing  {note}")
    for note in sorted(extra):
        print(f"  outside the grid  {note}")


if __name__ == "__main__":
    main()
