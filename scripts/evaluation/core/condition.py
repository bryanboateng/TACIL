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
            case block_push.Config(impulse_per_step=impulse_per_step):
                return f"block_push_{int(impulse_per_step)}"
            case action_noise.Config(noise_std=std):
                return f"action_noise_{int(std)}"


def build_conditions() -> list[EvaluationCondition]:
    """The evaluation suite: a clean reference, a sustained external shove (block shove),
    and per-step actuator error (action noise). Each disturbance is swept over a few
    strengths so every policy yields a reward-vs-strength curve.

    Single source of truth — consumed by both the evaluator and the renderer preview.

    NOTE: block push applies one impulse per control step across the window. At the
    default duration, the block moves ~impulse_per_step * 0.3 units while pushed, then
    stops. These magnitudes reproduce the previous shove distances (~[30, 60, 90] units).
    """
    block_shove_impulses = [1000.0, 200.0, 300.0]
    action_noise_stds = [5.0, 15.0, 30.0]

    return [
        # EvalCondition(name="clean", disturbance=None, bg_cmap="Greys"),
        # *(
        #     EvaluationCondition(
        #         disturbance=block_push.Config(
        #             impulse_per_step=impulse_per_step, step=100, duration=20
        #         ),
        #         color_map="Oranges",
        #     )
        #     for impulse_per_step in block_shove_impulses
        # )
        *(
            EvaluationCondition(
                disturbance=action_noise.Config(noise_std=std), color_map="Blues"
            )
            for std in action_noise_stds
        )
    ]
