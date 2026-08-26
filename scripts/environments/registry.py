from enum import StrEnum

from scripts.environments.environment import Environment


class EnvironmentName(StrEnum):
    PUSHT = "pusht"
    ROLL_BALL = "roll-ball"
    PUSH_CUBE = "push-cube"


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
