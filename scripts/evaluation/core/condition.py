from dataclasses import dataclass
from enum import Enum, auto
from itertools import chain


class FloorCondition(Enum):
    GRIPPY = auto()
    SLIPPERY = auto()


class DisturbanceLevel(Enum):
    LOW = auto()
    HIGH = auto()


@dataclass(frozen=True)
class Disturbance:
    level: DisturbanceLevel
    magnitude: float


@dataclass(frozen=True)
class Condition:
    floor_condition: FloorCondition
    disturbance: Disturbance | None

    @property
    def label(self) -> str:
        if self.disturbance is None:
            disturbance_label = "none"
        else:
            match self.disturbance.level:
                case DisturbanceLevel.LOW:
                    disturbance_label = "low"
                case DisturbanceLevel.HIGH:
                    disturbance_label = "high"
        return (
            f"damping_{self.floor_condition.name.lower()}"
            f"_disturbance_{disturbance_label}"
        )


def get_conditions() -> list[Condition]:
    return list(
        chain.from_iterable(
            _get_conditions_for_setting(
                floor_condition=floor_condition, disturbance_levels=disturbance_levels
            )
            for floor_condition, disturbance_levels in _get_settings().items()
        )
    )


# ---- Private --------------------------------------------


@dataclass(frozen=True)
class _DisturbanceLevels:
    low: float
    high: float


def _get_settings() -> dict[FloorCondition, _DisturbanceLevels]:
    return {
        FloorCondition.GRIPPY: _DisturbanceLevels(low=1_500, high=4_500),
        FloorCondition.SLIPPERY: _DisturbanceLevels(low=24, high=72),
    }


def _get_conditions_for_setting(
    *, floor_condition: FloorCondition, disturbance_levels: _DisturbanceLevels
) -> list[Condition]:
    disturbances: list[Disturbance | None] = [
        None,
        Disturbance(level=DisturbanceLevel.LOW, magnitude=disturbance_levels.low),
        Disturbance(level=DisturbanceLevel.HIGH, magnitude=disturbance_levels.high),
    ]
    return [
        Condition(floor_condition=floor_condition, disturbance=disturbance)
        for disturbance in disturbances
    ]
