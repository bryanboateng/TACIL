from dataclasses import dataclass
from enum import Enum, auto


class FloorCondition(Enum):
    """How well the moved object grips the ground.

    GRIPPY is the setting the demonstrations were recorded under.
    """

    GRIPPY = auto()
    SLIPPERY = auto()


class DisturbanceLevel(Enum):
    LOW = auto()
    HIGH = auto()


@dataclass(frozen=True)
class Condition:
    """One setting a policy is scored under.

    Every environment offers the same set of settings, so that results carry
    over from one task to the next. What each setting means in physics is up
    to the environment: a floor damping in one, a friction coefficient in the
    next.
    """

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
