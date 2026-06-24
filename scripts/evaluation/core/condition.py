from dataclasses import dataclass

from scripts.evaluation.core.disturbance import ActionNoise, BlockImpulse, Disturbance


@dataclass(frozen=True)
class EvalCondition:
    name: str
    disturbance: Disturbance | None
    bg_cmap: str = "Greys"


def build_conditions() -> list[EvalCondition]:
    """The evaluation suite: a clean reference, a one-shot external shove (block impulse),
    and per-step actuator error (action noise). Each disturbance is swept over a few
    strengths so every policy yields a reward-vs-strength curve.

    Single source of truth — consumed by both the evaluator and the renderer preview.

    NOTE: magnitudes are starting points and need calibration for PushT.
    """
    block_impulse_magnitudes = [10050.0]
    action_noise_stds = [5.0, 15.0, 30.0]

    return [
        EvalCondition(name="clean", disturbance=None, bg_cmap="Greys"),
        *(
            EvalCondition(
                name=f"block_impulse_{int(magnitude)}",
                disturbance=BlockImpulse(magnitude=magnitude),
                bg_cmap="Oranges",
            )
            for magnitude in block_impulse_magnitudes
        )
        * (
            EvalCondition(
                name=f"action_noise_{int(std)}",
                disturbance=ActionNoise(std=std),
                bg_cmap="Blues",
            )
            for std in action_noise_stds
        ),
    ]
