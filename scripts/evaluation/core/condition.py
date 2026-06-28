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
    """The evaluation suite: a clean reference, a sustained external shove (block shove),
    and per-step actuator error (action noise). Each disturbance is swept over a few
    strengths so every policy yields a reward-vs-strength curve.

    Single source of truth — consumed by both the evaluator and the renderer preview.

    NOTE: block push applies one impulse per control step across the window. Over the
    default window, the block moves ~strength * 0.3 units while pushed, then stops.
    These magnitudes reproduce the previous shove distances (~[30, 60, 90] units).
    """
    block_shove_strengths = [700.0]
    action_noise_fractions = [0.05, 0.15, 0.30]

    return [
        # EvalCondition(name="clean", disturbance=None, bg_cmap="Greys"),
        *(
            EvaluationCondition(
                disturbance=block_push.Config(
                    magnitude=strength, window=range(100, 120)
                ),
                color_map="Oranges",
            )
            for strength in block_shove_strengths
        )
        # *(
        #     EvaluationCondition(
        #         disturbance=action_noise.Config(noise_fraction=fraction), color_map="Blues"
        #     )
        #     for fraction in action_noise_fractions
        # )
    ]
