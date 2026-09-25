from wandb import Run

from scripts.shared import stage_names

STAGE_NAME = stage_names.POLICY

STEP_METRIC_NAME = f"{STAGE_NAME}/0_step"
EPOCH_METRIC_NAME = f"{STAGE_NAME}/0_epoch"

TRAINING_SECTION = f"{STAGE_NAME}/1_training"
VALIDATION_SECTION = f"{STAGE_NAME}/2_validation"


def define_step_metrics(*, run: Run, sections: tuple[str, ...]) -> None:
    run.define_metric(name=STEP_METRIC_NAME)
    for name in (EPOCH_METRIC_NAME, *(f"{section}/*" for section in sections)):
        run.define_metric(name=name, step_metric=STEP_METRIC_NAME)
