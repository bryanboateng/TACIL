from enum import StrEnum

from scripts.environments.environment import Environment


class EnvironmentName(StrEnum):
    PUSHT = "pusht"
    ROLL_BALL = "roll-ball"
    PUSH_CUBE = "push-cube"


# The wandb project holding an environment's demonstration artifacts. Each
# environment gets its own, because the artifacts inside a project are told
# apart by name alone: uploading a second environment under the shared name
# `dataset-100` would file it as a new version of the first one, and every
# `:latest` reference would then quietly resolve to the wrong task.
_DATASET_PROJECT_BY_NAME: dict[EnvironmentName, str] = {
    EnvironmentName.PUSHT: "datasets",
    EnvironmentName.ROLL_BALL: "datasets-rollball",
    EnvironmentName.PUSH_CUBE: "datasets-pushcube",
}


def dataset_project(*, name: EnvironmentName) -> str:
    return _DATASET_PROJECT_BY_NAME[name]


def dataset_artifact_name(*, name: EnvironmentName, percentage: int) -> str:
    """What one environment's demonstration artifact is called.

    The name has to carry the environment, because wandb caches a downloaded
    artifact under `artifacts/<name>:<version>` and ignores the project it
    came from. Two environments sharing the name `dataset-100` would therefore
    overwrite each other's copy on any machine that runs both.

    PushT keeps the plain names, so that its finished sweeps keep resolving.
    """
    prefix = "" if name is EnvironmentName.PUSHT else f"{name.value}-"
    return f"{prefix}dataset-{percentage}"


def create_environment(*, name: EnvironmentName) -> Environment:
    # The imports sit inside the function because each one pulls in a whole
    # simulator. A PushT run should not have to load ManiSkill, which needs a
    # graphics driver that is not on every machine.
    match name:
        case EnvironmentName.PUSHT:
            from scripts.environments.pusht.environment import PushTEnvironment

            return PushTEnvironment()
        case EnvironmentName.ROLL_BALL:
            from scripts.environments.maniskill.environment import ManiSkillEnvironment

            return ManiSkillEnvironment(task_id="RollBall-v1")
        case EnvironmentName.PUSH_CUBE:
            from scripts.environments.maniskill.environment import ManiSkillEnvironment

            return ManiSkillEnvironment(task_id="PushCube-v1")
