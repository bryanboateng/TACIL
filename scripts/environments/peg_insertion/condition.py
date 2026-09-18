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
    disturbance_impulse: float | None


SETTINGS_BY_CONDITION: dict[Condition, Settings] = {
    PegInsertionCondition(DisturbanceLevel.NONE): Settings(
        label="0_undisturbed", description="disturbance: none", disturbance_impulse=None
    ),
    PegInsertionCondition(DisturbanceLevel.LOW): Settings(
        label="1_low_disturbance",
        description="disturbance: low",
        disturbance_impulse=0.02,
    ),
    PegInsertionCondition(DisturbanceLevel.HIGH): Settings(
        label="2_high_disturbance",
        description="disturbance: high",
        disturbance_impulse=0.04,
    ),
}

UNDISTURBED_CONDITION: Condition = PegInsertionCondition(DisturbanceLevel.NONE)

ALL_CONDITIONS: tuple[Condition, ...] = (
    PegInsertionCondition(DisturbanceLevel.LOW),
    PegInsertionCondition(DisturbanceLevel.HIGH),
)
