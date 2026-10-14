from enum import StrEnum

from scripts.environments.environment import Environment


class EnvironmentName(StrEnum):
    PUSH_T = "PushT"
    PEG_INSERTION = "PegInsertionSide"
    STACK_CUBE = "StackCube"


def create_environment(*, name: EnvironmentName) -> Environment:
    match name:
        case EnvironmentName.PUSH_T:
            from scripts.environments.pusht.environment import PushTEnvironment

            return PushTEnvironment()
        case EnvironmentName.PEG_INSERTION:
            from scripts.environments.peg_insertion.environment import (
                PegInsertionEnvironment,
            )

            return PegInsertionEnvironment()
        case EnvironmentName.STACK_CUBE:
            from scripts.environments.stack_cube.environment import StackCubeEnvironment

            return StackCubeEnvironment()
