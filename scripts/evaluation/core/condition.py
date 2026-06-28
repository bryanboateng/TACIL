from dataclasses import dataclass

from scripts.evaluation.core.disturbance import action_noise, block_push


@dataclass(frozen=True)
class Undisturbed:
    pass


Condition = Undisturbed | block_push.Config | action_noise.Config


def label(condition: Condition) -> str:
    match condition:
        case Undisturbed():
            return "undisturbed"
        case block_push.Config(magnitude=magnitude, window=window):
            return f"block_push_{int(magnitude)}_{window.start}-{window.stop}"
        case action_noise.Config(noise_fraction=noise_fraction):
            return f"action_noise_{noise_fraction}"


def color_map(condition: Condition) -> str:
    match condition:
        case Undisturbed():
            return "Greys"
        case block_push.Config():
            return "Oranges"
        case action_noise.Config():
            return "Blues"


def build_conditions() -> list[Condition]:
    block_push_magnitudes = [700.0]
    action_noise_fractions = [0.05, 0.15, 0.30]

    return [
        Undisturbed(),
        *(
            block_push.Config(magnitude=magnitude, window=range(100, 120))
            for magnitude in block_push_magnitudes
        ),
        *(
            action_noise.Config(noise_fraction=noise_fraction)
            for noise_fraction in action_noise_fractions
        ),
    ]
