from dataclasses import dataclass
from enum import Enum, auto
from itertools import chain


class DisturbanceLevel(Enum):
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()


@dataclass(frozen=True)
class Disturbance:
    level: DisturbanceLevel
    magnitude: float


@dataclass(frozen=True)
class Condition:
    damping: float
    disturbance: Disturbance | None
    colormap: str

    @property
    def label(self) -> str:
        if self.disturbance is None:
            disturbance_label = "none"
        else:
            match self.disturbance.level:
                case DisturbanceLevel.LOW:
                    disturbance_label = "low"
                case DisturbanceLevel.MEDIUM:
                    disturbance_label = "medium"
                case DisturbanceLevel.HIGH:
                    disturbance_label = "high"
        return f"damping_{self.damping}_disturbance_{disturbance_label}"


def get_conditions() -> list[Condition]:
    return list(
        chain.from_iterable(
            _get_conditions_for_setting(setting=setting) for setting in _get_settings()
        )
    )


# ---- Private --------------------------------------------


@dataclass(frozen=True)
class _DisturbanceLevels:
    low: float
    medium: float
    high: float


@dataclass(frozen=True)
class _Setting:
    damping: float
    disturbance_levels: _DisturbanceLevels
    colormap: str


def _get_settings() -> list[_Setting]:
    return [
        _Setting(
            damping=0.0,
            disturbance_levels=_DisturbanceLevels(low=3_000, medium=6_000, high=12_000),
            colormap="Greys",
        ),
        _Setting(
            damping=0.2,
            disturbance_levels=_DisturbanceLevels(low=50, medium=100, high=200),
            colormap="Blues",
        ),
        _Setting(
            damping=0.4,
            disturbance_levels=_DisturbanceLevels(low=30, medium=60, high=120),
            colormap="Oranges",
        ),
    ]


def _get_conditions_for_setting(*, setting: _Setting) -> list[Condition]:
    levels = setting.disturbance_levels
    disturbances: list[Disturbance | None] = [
        None,
        Disturbance(level=DisturbanceLevel.LOW, magnitude=levels.low),
        Disturbance(level=DisturbanceLevel.MEDIUM, magnitude=levels.medium),
        Disturbance(level=DisturbanceLevel.HIGH, magnitude=levels.high),
    ]
    return [
        Condition(
            damping=setting.damping, disturbance=disturbance, colormap=setting.colormap
        )
        for disturbance in disturbances
    ]
