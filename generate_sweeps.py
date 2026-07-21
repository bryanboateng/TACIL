"""Generate the DP-training and evaluation sweep YAMLs from W&B run history.

Each sweep varies one ``spec`` value of the form ``<notes>::<artifact>``. A POSIX
sh shim (``sweep_run_dp.sh`` / ``sweep_run_eval.sh``) is the sweep's program: it
splits the spec and calls the untouched ``train.py`` / ``evaluate_policy.py`` with
the notes supplied from the outside. So the note is set externally, the python
scripts stay pure, and there is no python wrapper. (sh, so it runs on any agent
machine without fish installed.)

The note is derived from the producing run's note and threaded down the lineage:

    phase 1 note   data-lip-1-action-0.1-error-0.5   (create-tacil-dataset run)
    phase 2 note   policy-lip-1-action-0.1-error-0.5 (train-diffusion-policy run)
    phase 3 note   eval-lip-1-action-0.1-error-0.5   (evaluate-policy run)

    # after phase 1 (datasets exist):
    uv run python generate_sweeps.py dp    # -> dp_sweep.yaml
    # after phase 2 (policies exist):
    uv run python generate_sweeps.py eval  # -> eval_sweep.yaml

Then:  wandb sweep -p opq dp_sweep.yaml   and   wandb agent <sweep-id>   per GPU.
"""

import argparse

import wandb
import yaml

SPEC_SEPARATOR = "::"


def _dp_consumer_done(run) -> bool:
    return any(a.type == "model" for a in run.logged_artifacts())


def _eval_consumer_done(run) -> bool:
    return any(str(key).endswith("/reward_mean") for key in run.summary.keys())


# Per phase: which producing runs to read, which artifact they emit, how to
# rewrite their note, which fish shim runs the real script, and how to tell
# whether a combination was already produced (so it can be skipped).
PHASES = {
    "dp": {
        "source_job_type": "create-tacil-dataset",
        "artifact_type": "dataset",
        "note_prefix_from": "data-",
        "note_prefix_to": "policy-",
        "consumer_job_type": "train-diffusion-policy",
        "consumer_done": _dp_consumer_done,
        "program": "sweep_run_dp.sh",
        "output": "dp_sweep.yaml",
        "metric": "selected/reward_mean",
    },
    "eval": {
        "source_job_type": "train-diffusion-policy",
        "artifact_type": "model",
        "note_prefix_from": "policy-",
        "note_prefix_to": "eval-",
        "consumer_job_type": "evaluate-policy",
        "consumer_done": _eval_consumer_done,
        "program": "sweep_run_eval.sh",
        "output": "eval_sweep.yaml",
        "metric": None,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=sorted(PHASES))
    parser.add_argument("--project", default="opq")
    parser.add_argument("--entity", default=None)
    parser.add_argument(
        "--include-done",
        action="store_true",
        help="Also include combinations already produced by a successful "
        "downstream run (default skips them so only new work is scheduled).",
    )
    arguments = parser.parse_args()

    phase = PHASES[arguments.phase]
    api = wandb.Api()
    entity = arguments.entity or api.default_entity
    run_path = f"{entity}/{arguments.project}"

    completed = (
        set()
        if arguments.include_done
        else _completed_downstream_notes(api=api, run_path=run_path, phase=phase)
    )
    if completed:
        print(f"Skipping {len(completed)} combinations already done.")

    specs = _build_specs(
        api=api, run_path=run_path, phase=phase, completed=completed
    )
    if not specs:
        raise SystemExit(
            "Nothing to schedule: every combination is already done "
            "(use --include-done to rebuild), or no matching source runs exist "
            f"in {run_path}."
        )

    _write_sweep(
        output_path=phase["output"],
        program=phase["program"],
        project=arguments.project,
        metric=phase["metric"],
        specs=specs,
    )
    print(f"Wrote {phase['output']} with {len(specs)} runs.")


def _completed_downstream_notes(
    *, api: wandb.Api, run_path: str, phase: dict
) -> set[str]:
    # Downstream notes whose consumer run already produced its output, so the
    # combination does not need to be scheduled again. A crashed/incomplete
    # consumer fails the check and is therefore re-scheduled.
    done: set[str] = set()
    for run in api.runs(path=run_path):
        if run.job_type != phase["consumer_job_type"]:
            continue
        note = (run.notes or "").strip()
        if note.startswith(phase["note_prefix_to"]) and phase["consumer_done"](run):
            done.add(note)
    return done


def _build_specs(
    *, api: wandb.Api, run_path: str, phase: dict, completed: set[str]
) -> list[str]:
    specs: list[str] = []
    seen: set[str] = set()
    for run in api.runs(path=run_path):
        if run.job_type != phase["source_job_type"]:
            continue
        note = (run.notes or "").strip()
        if not note.startswith(phase["note_prefix_from"]):
            continue
        downstream_note = (
            phase["note_prefix_to"] + note[len(phase["note_prefix_from"]) :]
        )
        # Skip combinations already done, and keep only one source per combination.
        if downstream_note in completed or downstream_note in seen:
            continue
        artifact = next(
            (a for a in run.logged_artifacts() if a.type == phase["artifact_type"]),
            None,
        )
        if artifact is None:
            continue
        seen.add(downstream_note)
        specs.append(f"{downstream_note}{SPEC_SEPARATOR}{artifact.name}")
    return sorted(specs)


def _write_sweep(
    *,
    output_path: str,
    program: str,
    project: str,
    metric: str | None,
    specs: list[str],
) -> None:
    sweep = {
        "program": program,
        "method": "grid",
        "parameters": {"spec": {"values": specs}},
        # Pass the project to the shim so runs land in the same project we read.
        "command": ["sh", program, project, "${args}"],
    }
    if metric is not None:
        sweep["metric"] = {"name": metric, "goal": "maximize"}

    with open(output_path, "w") as output_file:
        yaml.safe_dump(sweep, output_file, sort_keys=False, default_flow_style=False)


if __name__ == "__main__":
    main()
