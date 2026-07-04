from dataclasses import dataclass
from enum import Enum


class DampingLevel(float, Enum):
    NONE = 0.0  # quasi-static default floor
    LOW = 0.2
    HIGH = 0.4  # slidiest floor


@dataclass(frozen=True)
class Condition:
    damping: DampingLevel
    disturbance_magnitude: int | None


def label(condition: Condition) -> str:
    push_part = (
        "None"
        if condition.disturbance_magnitude is None
        else f"{condition.disturbance_magnitude}"
    )
    return f"da_{condition.damping.name}+di_{push_part}"


def condition_colormap(*, damping_level: DampingLevel) -> str:
    # Each damping level renders in its own single-hue palette so the floor can be read
    # straight off a video. RecoloredRenderWrapper samples the named colormap across its range.
    match damping_level:
        case DampingLevel.NONE:
            return "Blues"
        case DampingLevel.LOW:
            return "Reds"
        case DampingLevel.HIGH:
            return "Greens"
        case _:
            raise ValueError(f"No colormap for damping level {damping_level!r}")


def build_conditions() -> list[Condition]:
    # The disturbance magnitude is the raw force on the block, and the same force acts very
    # differently per floor (a damping-0 floor eats most of it, a slidy floor amplifies it),
    # so each damping level gets its own hand-picked magnitudes. Within one damping the
    # effect is linear in the magnitude, so levels are easy to stage; magnitudes are not
    # comparable across damping levels, but policies are only compared within a condition.
    # Reference (from-rest burst peak): NONE@6000 -> ~60 px/s, LOW@100 -> ~50, HIGH@60 -> ~40.
    magnitudes_by_damping: dict[DampingLevel, list[int]] = {
        DampingLevel.NONE: [6000],
        DampingLevel.LOW: [100],
        DampingLevel.HIGH: [60],
    }
    return [
        Condition(damping=damping, disturbance_magnitude=disturbance_magnitude)
        for damping, disturbance_magnitudes in magnitudes_by_damping.items()
        for disturbance_magnitude in disturbance_magnitudes
    ]
