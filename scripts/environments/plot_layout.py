from dataclasses import dataclass


@dataclass(frozen=True)
class PositionGroup:
    label: str
    entries: slice


@dataclass(frozen=True)
class OrientationGroup:
    label: str
    index: int


@dataclass(frozen=True)
class PositionUnit:
    label: str
    scale: float


@dataclass(frozen=True)
class PlotLayout:
    position_groups: tuple[PositionGroup, ...]
    orientation_groups: tuple[OrientationGroup, ...]
    horizontal_limits: tuple[float, float]
    vertical_limits: tuple[float, float]
    position_unit: PositionUnit
