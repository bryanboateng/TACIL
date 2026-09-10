from dataclasses import dataclass


@dataclass(frozen=True)
class PositionGroup:
    """Two state entries that name a place on the plane."""

    label: str
    entries: slice


@dataclass(frozen=True)
class StatePicture:
    """What a diagnostic plot can show of a state.

    A state is a long vector, and most of its entries mean nothing on paper.
    These are the ones that do: the places the moved things occupy, and the
    one angle, where the environment has one.
    """

    position_groups: tuple[PositionGroup, ...]
    horizontal_limits: tuple[float, float]
    vertical_limits: tuple[float, float]
    position_unit: str
    orientation_index: int | None
