import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass

import tyro
import wandb
import wandb.util
from rich.pretty import pretty_repr
from wandb import Run
from wandb.apis.public import Run as FinishedRun

from scripts.evaluation.core.condition import Condition
from scripts.shared.log import create_logger
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH
from scripts.sweep.config import Config

logger = create_logger()

# Set by the sweep agent for the trial run itself. A stage must not inherit them:
# the run and sweep ids would make it be adopted by the sweep and reported as one
# of its trials, the rest would label it as the trial. It gets its own values in
# _stage_environment instead.
SWEEP_ENVIRONMENT_VARIABLE_NAMES = (
    "WANDB_RUN_ID",
    "WANDB_SWEEP_ID",
    "WANDB_SWEEP_PARAM_PATH",
    "WANDB_RUN_GROUP",
    "WANDB_JOB_TYPE",
    "WANDB_NAME",
    "WANDB_NOTES",
    "WANDB_TAGS",
    "WANDB_RESUME",
    "WANDB_SERVICE",
)

# Stage diagnostics worth having on the trial run, so a single row of the sweep
# table explains why a trial scored the way it did.
FORWARDED_SUMMARY_KEYS = {
    "train-tacil-dynamics": {
        "checkpoint/validation_loss_mean": "dynamics/validation_loss_mean",
        "error_model/uncentered_r_squared": "dynamics/error_model_r_squared",
    },
    "create-tacil-dataset": {
        "2_rollout/accepted_anchor_fraction": "data/accepted_anchor_fraction",
        "4_augmented_episodes/synthetic_part_length_mean": "data/synthetic_part_length_mean",
        "5_combined_dataset/synthetic_transition_fraction": "data/synthetic_transition_fraction",
    },
    "train-diffusion-policy": {
        "selected/epoch": "policy/selected_epoch",
        "selected/reward_mean": "policy/selected_reward_mean",
    },
}


# How long a stage may take to shut down on its own before it is killed.
STAGE_SHUTDOWN_TIMEOUT_SECONDS = 30

# The exit status a shell reports for a process ended by an interrupt.
INTERRUPTED_EXIT_CODE = 130


def main() -> None:
    config = tyro.cli(f=Config, config=[tyro.conf.UsePythonSyntaxForLiteralCollections])
    logger.info(pretty_repr(config))
    # Take the same route out for a signalled shutdown as for Ctrl-C, so the
    # running stage is stopped either way.
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    try:
        _run(config=config)
    except KeyboardInterrupt:
        logger.warning("Interrupted. The trial is incomplete.")
        sys.exit(INTERRUPTED_EXIT_CODE)


def _raise_keyboard_interrupt(signal_number: int, _frame: object) -> None:
    raise KeyboardInterrupt(f"Received signal {signal_number}.")


@dataclass(frozen=True)
class Notes:
    dynamics: str
    data: str
    policy: str
    evaluation: str
    trial: str


def _run(*, config: Config) -> None:
    condition = Condition(
        floor_condition=config.floor_condition,
        disturbance_level=config.disturbance_level,
    )
    notes = _build_notes(config=config, condition=condition)

    # The agent has already chosen the trial's run id; outside a sweep there is
    # none, so choose one here. Either way it is known before the run starts and
    # can therefore name the group that ties the stages to this trial.
    trial_run_id = os.environ.get("WANDB_RUN_ID") or wandb.util.generate_id()

    with wandb.init(
        id=trial_run_id,
        group=trial_run_id,
        project=config.wandb.project,
        notes=notes.trial,
        job_type="tacil-pipeline",
        config=asdict(config),
        dir=OUTPUT_DIRECTORY_PATH,
    ) as trial_run:
        api = wandb.Api()
        run_path = f"{trial_run.entity}/{trial_run.project}"

        dynamics_run = _run_stage(
            api=api,
            run_path=run_path,
            group=trial_run_id,
            job_type="train-tacil-dynamics",
            note=notes.dynamics,
            command=[
                "-m",
                "scripts.data.augmentation.tacil.create_dynamics"
                ".train_dynamics_model_ensemble",
                f"--wandb.project={config.wandb.project}",
                f"--wandb.notes={notes.dynamics}",
                f"--data.artifact={config.expert_dataset_artifact}",
                f"--data.loader.window-length={config.window_length}",
                "--architecture.lipschitz-constant-upper-bound="
                f"{_lipschitz_argument(config=config)}",
                f"--seed={config.seed}",
            ],
        )

        data_run = _run_stage(
            api=api,
            run_path=run_path,
            group=trial_run_id,
            job_type="create-tacil-dataset",
            note=notes.data,
            command=[
                "-m",
                "scripts.data.augmentation.tacil.create_data.create_augmented_data",
                f"--wandb.project={config.wandb.project}",
                f"--wandb.notes={notes.data}",
                "--model-artifact="
                f"{_artifact_name(run=dynamics_run, artifact_type='model')}",
                f"--dataset-artifact={config.expert_dataset_artifact}",
                f"--action-noise-fraction={config.action_noise_fraction}",
                f"--error-limit-fraction={config.error_limit_fraction}",
                f"--seed={config.seed}",
            ],
        )

        policy_run = _run_stage(
            api=api,
            run_path=run_path,
            group=trial_run_id,
            job_type="train-diffusion-policy",
            note=notes.policy,
            command=[
                "-m",
                "scripts.policies.dp.train",
                f"--wandb.project={config.wandb.project}",
                f"--wandb.notes={notes.policy}",
                "--dataset-artifact="
                f"{_artifact_name(run=data_run, artifact_type='dataset')}",
                f"--evaluation.floor-condition={config.floor_condition.name}",
                "--evaluation.disturbance-level="
                f"{_disturbance_level_argument(config=config)}",
                f"--seed={config.seed}",
            ],
        )

        evaluation_run = _run_stage(
            api=api,
            run_path=run_path,
            group=trial_run_id,
            job_type="evaluate-policy",
            note=notes.evaluation,
            command=[
                "-m",
                "scripts.evaluation.evaluate_policy",
                f"--wandb.project={config.wandb.project}",
                f"--wandb.notes={notes.evaluation}",
                f"--floor-condition={config.floor_condition.name}",
                f"--disturbance-level={_disturbance_level_argument(config=config)}",
                "dp",
                "--checkpoint-artifact="
                f"{_artifact_name(run=policy_run, artifact_type='model')}",
            ],
        )

        for stage_name, stage_job_type, stage_run in (
            ("dynamics", "train-tacil-dynamics", dynamics_run),
            ("data", "create-tacil-dataset", data_run),
            ("policy", "train-diffusion-policy", policy_run),
            ("evaluation", "evaluate-policy", evaluation_run),
        ):
            trial_run.summary[f"stage_run/{stage_name}"] = stage_run.name
            _forward_summary(
                source=stage_run,
                keys=FORWARDED_SUMMARY_KEYS.get(stage_job_type, {}),
                target=trial_run,
            )

        # What the sweep maximizes.
        reward_key = f"{condition.label}/reward_mean"
        _forward_summary(
            source=evaluation_run,
            keys={
                f"{condition.label}/{statistic}": f"{condition.label}/{statistic}"
                for statistic in ("reward_mean", "reward_std", "success_rate")
            },
            target=trial_run,
        )
        logger.info(
            "Trial finished: %s = %s", reward_key, trial_run.summary[reward_key]
        )


def _build_notes(*, config: Config, condition: Condition) -> Notes:
    dynamics = (
        f"data={config.expert_dataset_artifact}"
        f"|lip={config.lipschitz_constant_upper_bound}"
        f"|win={config.window_length}"
        f"|seed={config.seed}"
    )
    data = (
        f"{dynamics}"
        f"|noise={config.action_noise_fraction:.4f}"
        f"|err={config.error_limit_fraction:.4f}"
    )
    policy = f"{data}|cond={condition.label}"
    return Notes(
        dynamics=f"dyn|{dynamics}",
        data=f"data|{data}",
        policy=f"policy|{policy}",
        evaluation=f"eval|{policy}",
        trial=f"trial|{policy}",
    )


def _disturbance_level_argument(*, config: Config) -> str:
    level = config.disturbance_level
    return "None" if level is None else level.name


def _lipschitz_argument(*, config: Config) -> str:
    value = config.lipschitz_constant_upper_bound
    if value.lower() == "none":
        return "None"
    # Reject anything unparseable here rather than in the stage, where it would
    # cost a queued job to find out.
    return str(float(value))


def _run_stage(
    *,
    api: wandb.Api,
    run_path: str,
    group: str,
    job_type: str,
    note: str,
    command: list[str],
) -> FinishedRun:
    reusable_run = _find_reusable_run(
        api=api, run_path=run_path, job_type=job_type, note=note
    )
    if reusable_run is not None:
        logger.info("Reusing %s for %s (%s).", reusable_run.name, job_type, note)
        return reusable_run

    run_id = wandb.util.generate_id()
    logger.info("Running %s as %s (%s).", job_type, run_id, note)
    _run_command(
        command=command,
        environment=_stage_environment(
            run_id=run_id, group=group, name=f"{group}-{job_type}"
        ),
    )
    return _wait_for_run(api=api, run_path=f"{run_path}/{run_id}")


def _run_command(*, command: list[str], environment: dict[str, str]) -> None:
    # The stage gets a session of its own, so an interrupt typed in the terminal
    # reaches this process alone. It is passed on below, once, and the stage is
    # given the chance to end its own run before it is killed.
    process = subprocess.Popen(
        args=[sys.executable, *command], env=environment, start_new_session=True
    )
    try:
        return_code = process.wait()
    except KeyboardInterrupt:
        _stop_process(process=process)
        raise
    if return_code != 0:
        raise subprocess.CalledProcessError(returncode=return_code, cmd=process.args)


def _stop_process(*, process: subprocess.Popen) -> None:
    logger.warning("Stopping stage %d.", process.pid)
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=STAGE_SHUTDOWN_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        # Kill the whole group: the stage itself may be wedged, and its data
        # loaders and renderers are children of it.
        logger.warning("Stage %d did not stop. Killing it.", process.pid)
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _find_reusable_run(
    *, api: wandb.Api, run_path: str, job_type: str, note: str
) -> FinishedRun | None:
    return next(
        iter(
            api.runs(
                path=run_path,
                filters={"jobType": job_type, "notes": note, "state": "finished"},
            )
        ),
        None,
    )


def _wait_for_run(*, api: wandb.Api, run_path: str) -> FinishedRun:
    # The stage has exited, but the backend may need a moment before the run and
    # its artifact are readable.
    run = api.run(path=run_path)
    for attempt in range(10):
        if run.state == "finished":
            return run
        time.sleep(3)
        # The Api caches what it has already fetched, so ask for it again.
        run.load(force=True)
        logger.info(
            "Waiting for %s to be readable (attempt %d).", run_path, attempt + 1
        )
    raise RuntimeError(f"{run_path} exited successfully but never became readable.")


def _stage_environment(*, run_id: str, group: str, name: str) -> dict[str, str]:
    environment = {
        variable_name: value
        for variable_name, value in os.environ.items()
        if variable_name not in SWEEP_ENVIRONMENT_VARIABLE_NAMES
    }
    # Choose the stage's run id here, so its artifact can be looked up by id
    # rather than by parsing a generated run name.
    environment["WANDB_RUN_ID"] = run_id
    # The stage stays outside the sweep, so the group and the name are all that
    # tie it to the trial it was run for.
    environment["WANDB_RUN_GROUP"] = group
    environment["WANDB_NAME"] = name
    sweep_id = os.environ.get("WANDB_SWEEP_ID")
    if sweep_id is not None:
        environment["WANDB_TAGS"] = f"sweep-{sweep_id}"
    return environment


def _artifact_name(*, run: FinishedRun, artifact_type: str) -> str:
    artifact = next(
        (
            artifact
            for artifact in run.logged_artifacts()
            if artifact.type == artifact_type
        ),
        None,
    )
    if artifact is None:
        raise RuntimeError(
            f"Run {run.name} logged no artifact of type {artifact_type!r}."
        )
    return artifact.name


def _forward_summary(*, source: FinishedRun, keys: dict[str, str], target: Run) -> None:
    for source_key, target_key in keys.items():
        if source_key in source.summary:
            target.summary[target_key] = source.summary[source_key]


if __name__ == "__main__":
    main()
