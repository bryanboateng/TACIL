from dataclasses import dataclass

from scripts.evaluation.core.disturbance import action_noise, block_push

Disturbance = block_push.Config | action_noise.Config


@dataclass(frozen=True)
class EvaluationCondition:
    disturbance: Disturbance | None
    color_map: str

    def __str__(self) -> str:
        match self.disturbance:
            case None:
                return "clean"
            case block_push.Config(magnitude=strength):
                return f"block_push_{int(strength)}"
            case action_noise.Config(noise_fraction=fraction):
                return f"action_noise_{fraction}"


def build_conditions() -> list[EvaluationCondition]:
    block_push_magnitudes = [700.0]
    action_noise_fractions = [0.05, 0.15, 0.30]

    return [
        EvaluationCondition(disturbance=None, color_map="Greys"),
        *(
            EvaluationCondition(
                disturbance=block_push.Config(
                    magnitude=magnitude, window=range(100, 120)
                ),
                color_map="Oranges",
            )
            for magnitude in block_push_magnitudes
        ),
        *(
            EvaluationCondition(
                disturbance=action_noise.Config(noise_fraction=noise_fraction),
                color_map="Blues",
            )
            for noise_fraction in action_noise_fractions
        ),
    ]
