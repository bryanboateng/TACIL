from enum import StrEnum

from scripts.environments.environment import Environment


class EnvironmentName(StrEnum):
    PUSHT = "PushT"
    PEG_INSERTION = "PegInsertionSide"


def create_environment(*, name: EnvironmentName) -> Environment:
    match name:
        case EnvironmentName.PUSHT:
            from scripts.environments.pusht.environment import PushTEnvironment

            return PushTEnvironment()
        case EnvironmentName.PEG_INSERTION:
            from scripts.environments.peg_insertion.environment import (
                PegInsertionEnvironment,
            )

            return PegInsertionEnvironment()
