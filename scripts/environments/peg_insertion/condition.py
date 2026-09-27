from dataclasses import dataclass
from enum import Enum, auto

from scripts.environments.condition import Condition


class DisturbanceLevel(Enum):
    NONE = auto()
    LOW = auto()
    HIGH = auto()


@dataclass(frozen=True)
class PegInsertionCondition:
    disturbance_level: DisturbanceLevel

    @property
    def label(self) -> str:
        return SETTINGS_BY_CONDITION[self].label


@dataclass(frozen=True)
class Settings:
    label: str
    description: str
    disturbance_shift: float | None


SETTINGS_BY_CONDITION: dict[Condition, Settings] = {
    PegInsertionCondition(DisturbanceLevel.NONE): Settings(
        label="1_undisturbed", description="disturbance: none", disturbance_shift=None
    ),
    PegInsertionCondition(DisturbanceLevel.LOW): Settings(
        label="2_low_disturbance",
        description="disturbance: low",
        disturbance_shift=0.01,
    ),
    PegInsertionCondition(DisturbanceLevel.HIGH): Settings(
        label="3_high_disturbance",
        description="disturbance: high",
        disturbance_shift=0.02,
    ),
}

UNDISTURBED_CONDITION: Condition = PegInsertionCondition(DisturbanceLevel.NONE)

ALL_CONDITIONS: tuple[Condition, ...] = tuple(SETTINGS_BY_CONDITION)
