from dataclasses import dataclass

from scripts.evaluation.core.disturbance import block_push


@dataclass(frozen=True)
class Undisturbed:
    pass


Condition = Undisturbed | block_push.Config


def label(condition: Condition) -> str:
    match condition:
        case Undisturbed():
            return "undisturbed"
        case block_push.Config(magnitude=magnitude, period=period, duration=duration):
            return f"block_push_{int(magnitude)}_p{period}_d{duration}"


def color_map(condition: Condition) -> str:
    match condition:
        case Undisturbed():
            return "Greys"
        case block_push.Config():
            return "Oranges"


def build_conditions() -> list[Condition]:
    block_push_magnitudes = [500.0]

    return [
        # Undisturbed(),
        *(
            block_push.Config(magnitude=magnitude, period=50, duration=10)
            for magnitude in block_push_magnitudes
        ),
    ]
