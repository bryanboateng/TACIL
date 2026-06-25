from dataclasses import dataclass

from scripts.evaluation.core.disturbance import ActionNoise, BlockShove, Disturbance


@dataclass(frozen=True)
class EvalCondition:
    name: str
    disturbance: Disturbance | None
    bg_cmap: str = "Greys"


def build_conditions() -> list[EvalCondition]:
    """The evaluation suite: a clean reference, a sustained external shove (block shove),
    and per-step actuator error (action noise). Each disturbance is swept over a few
    strengths so every policy yields a reward-vs-strength curve.

    Single source of truth — consumed by both the evaluator and the renderer preview.

    NOTE: a shove force at the default duration moves the block ~force * 0.003 units
    while it is pushed, then it stops.
    """
    block_shove_forces = [10000.0, 20000.0, 30000.0]
    action_noise_stds = [5.0, 15.0, 30.0]

    return [
        # EvalCondition(name="clean", disturbance=None, bg_cmap="Greys"),
        *(
            EvalCondition(
                name=f"block_shove_{int(force)}",
                disturbance=BlockShove(force=force),
                bg_cmap="Oranges",
            )
            for force in block_shove_forces
        )
        # * (
        #     EvalCondition(
        #         name=f"action_noise_{int(std)}",
        #         disturbance=ActionNoise(std=std),
        #         bg_cmap="Blues",
        #     )
        #     for std in action_noise_stds
        # ),
    ]
