from dataclasses import dataclass
from enum import Enum, auto

from scripts.environments.condition import Condition


class FloorCondition(Enum):
    GRIPPY = auto()
    SLIPPERY = auto()


class DisturbanceLevel(Enum):
    LOW = auto()
    HIGH = auto()


@dataclass(frozen=True)
class PushTCondition:
    floor_condition: FloorCondition
    disturbance_level: DisturbanceLevel | None

    @property
    def label(self) -> str:
        return SETTINGS_BY_CONDITION[self].label


@dataclass(frozen=True)
class Settings:
    label: str
    description: str
    damping: float
    colormap: str
    disturbance_magnitude: float | None


SETTINGS_BY_CONDITION: dict[Condition, Settings] = {
    PushTCondition(
        floor_condition=FloorCondition.GRIPPY, disturbance_level=None
    ): Settings(
        label="1_grippy_undisturbed",
        description="damping: 0.0   disturbance: none",
        damping=0.0,
        colormap="Blues",
        disturbance_magnitude=None,
    ),
    PushTCondition(
        floor_condition=FloorCondition.GRIPPY, disturbance_level=DisturbanceLevel.LOW
    ): Settings(
        label="2_grippy_low_disturbance",
        description="damping: 0.0   disturbance: low",
        damping=0.0,
        colormap="Blues",
        disturbance_magnitude=1_500.0,
    ),
    PushTCondition(
        floor_condition=FloorCondition.GRIPPY, disturbance_level=DisturbanceLevel.HIGH
    ): Settings(
        label="3_grippy_high_disturbance",
        description="damping: 0.0   disturbance: high",
        damping=0.0,
        colormap="Blues",
        disturbance_magnitude=4_500.0,
    ),
    PushTCondition(
        floor_condition=FloorCondition.SLIPPERY, disturbance_level=None
    ): Settings(
        label="4_slippery_undisturbed",
        description="damping: 0.2   disturbance: none",
        damping=0.2,
        colormap="Oranges",
        disturbance_magnitude=None,
    ),
    PushTCondition(
        floor_condition=FloorCondition.SLIPPERY, disturbance_level=DisturbanceLevel.LOW
    ): Settings(
        label="5_slippery_low_disturbance",
        description="damping: 0.2   disturbance: low",
        damping=0.2,
        colormap="Oranges",
        disturbance_magnitude=24.0,
    ),
    PushTCondition(
        floor_condition=FloorCondition.SLIPPERY, disturbance_level=DisturbanceLevel.HIGH
    ): Settings(
        label="6_slippery_high_disturbance",
        description="damping: 0.2   disturbance: high",
        damping=0.2,
        colormap="Oranges",
        disturbance_magnitude=72.0,
    ),
}

ALL_CONDITIONS: tuple[Condition, ...] = tuple(SETTINGS_BY_CONDITION)
