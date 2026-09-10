from enum import StrEnum

from scripts.environments.environment import Environment


class EnvironmentName(StrEnum):
    PUSHT = "pusht"
    PUSH_CUBE = "push-cube"


def dataset_artifact_name(*, environment: EnvironmentName, episode_count: int) -> str:
    """What one environment's demonstration artifact is called.

    The name has to carry the environment, because wandb caches a downloaded
    artifact under `artifacts/<name>:<version>` and ignores the project it
    came from. Two environments sharing the name `dataset-100` would therefore
    overwrite each other's copy on any machine that runs both.

    PushT keeps the plain names, so that its finished sweeps keep resolving.
    """
    prefix = "" if environment is EnvironmentName.PUSHT else f"{environment.value}-"
    return f"{prefix}dataset-{episode_count}"


def create_environment(*, environment: EnvironmentName) -> Environment:
    # The imports sit inside the function because each one pulls in a whole
    # simulator. A PushT run should not have to load ManiSkill, which needs a
    # graphics driver that is not on every machine.
    match environment:
        case EnvironmentName.PUSHT:
            from scripts.environments.pusht.environment import PushTEnvironment

            return PushTEnvironment()
        case EnvironmentName.PUSH_CUBE:
            from scripts.environments.maniskill.environment import ManiSkillEnvironment

            return ManiSkillEnvironment(task_id="PushCube-v1")
