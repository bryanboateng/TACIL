from dataclasses import dataclass
from enum import Enum, auto


class FloorCondition(Enum):
    GRIPPY = auto()
    SLIPPERY = auto()


class DisturbanceLevel(Enum):
    LOW = auto()
    HIGH = auto()


@dataclass(frozen=True)
class Condition:
    floor_condition: FloorCondition
    disturbance_level: DisturbanceLevel | None

    @property
    def label(self) -> str:
        return _LABELS[self]


_LABELS: dict[Condition, str] = {
    Condition(FloorCondition.GRIPPY, None): "1_grippy_undisturbed",
    Condition(FloorCondition.GRIPPY, DisturbanceLevel.LOW): "2_grippy_low_disturbance",
    Condition(FloorCondition.GRIPPY, DisturbanceLevel.HIGH): (
        "3_grippy_high_disturbance"
    ),
    Condition(FloorCondition.SLIPPERY, None): "4_slippery_undisturbed",
    Condition(FloorCondition.SLIPPERY, DisturbanceLevel.LOW): (
        "5_slippery_low_disturbance"
    ),
    Condition(FloorCondition.SLIPPERY, DisturbanceLevel.HIGH): (
        "6_slippery_high_disturbance"
    ),
}

ALL_CONDITIONS: tuple[Condition, ...] = tuple(_LABELS)
