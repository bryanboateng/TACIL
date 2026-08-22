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
        if self.disturbance_level is None:
            disturbance_label = "undisturbed"
        else:
            match self.disturbance_level:
                case DisturbanceLevel.LOW:
                    disturbance_label = "low_disturbance"
                case DisturbanceLevel.HIGH:
                    disturbance_label = "high_disturbance"
        return f"{self.floor_condition.name.lower()}_{disturbance_label}"


ALL_CONDITIONS: tuple[Condition, ...] = tuple(
    Condition(floor_condition=floor_condition, disturbance_level=disturbance_level)
    for floor_condition in FloorCondition
    for disturbance_level in (None, DisturbanceLevel.LOW, DisturbanceLevel.HIGH)
)

VALIDATION_CONDITION = Condition(
    floor_condition=FloorCondition.GRIPPY, disturbance_level=DisturbanceLevel.LOW
)
