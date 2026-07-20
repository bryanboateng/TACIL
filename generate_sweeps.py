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

# Per phase: which producing runs to read, which artifact they emit, how to
# rewrite their note for this phase, and which fish shim runs the real script.
PHASES = {
    "dp": {
        "source_job_type": "create-tacil-dataset",
        "artifact_type": "dataset",
        "note_prefix_from": "data-",
        "note_prefix_to": "policy-",
        "program": "sweep_run_dp.sh",
        "output": "dp_sweep.yaml",
        "metric": "validation/reward_mean",
    },
    "eval": {
        "source_job_type": "train-diffusion-policy",
        "artifact_type": "model",
        "note_prefix_from": "policy-",
        "note_prefix_to": "eval-",
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
    arguments = parser.parse_args()

    phase = PHASES[arguments.phase]
    api = wandb.Api()
    entity = arguments.entity or api.default_entity

    specs = _build_specs(
        api=api, run_path=f"{entity}/{arguments.project}", phase=phase
    )
    if not specs:
        raise SystemExit(
            f"No '{phase['source_job_type']}' runs with a '{phase['note_prefix_from']}' "
            f"note and a '{phase['artifact_type']}' artifact found in "
            f"{entity}/{arguments.project}."
        )

    _write_sweep(
        output_path=phase["output"],
        program=phase["program"],
        metric=phase["metric"],
        specs=specs,
    )
    print(f"Wrote {phase['output']} with {len(specs)} runs.")


def _build_specs(*, api: wandb.Api, run_path: str, phase: dict) -> list[str]:
    specs: list[str] = []
    for run in api.runs(path=run_path):
        if run.job_type != phase["source_job_type"]:
            continue
        note = (run.notes or "").strip()
        if not note.startswith(phase["note_prefix_from"]):
            continue
        artifact = next(
            (a for a in run.logged_artifacts() if a.type == phase["artifact_type"]),
            None,
        )
        if artifact is None:
            continue
        downstream_note = (
            phase["note_prefix_to"] + note[len(phase["note_prefix_from"]) :]
        )
        specs.append(f"{downstream_note}{SPEC_SEPARATOR}{artifact.name}")
    return sorted(specs)


def _write_sweep(
    *, output_path: str, program: str, metric: str | None, specs: list[str]
) -> None:
    sweep = {
        "program": program,
        "method": "grid",
        "parameters": {"spec": {"values": specs}},
        "command": ["sh", program, "${args}"],
    }
    if metric is not None:
        sweep["metric"] = {"name": metric, "goal": "maximize"}

    with open(output_path, "w") as output_file:
        yaml.safe_dump(sweep, output_file, sort_keys=False, default_flow_style=False)


if __name__ == "__main__":
    main()
