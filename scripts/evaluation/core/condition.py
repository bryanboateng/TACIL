from dataclasses import dataclass

from scripts.evaluation.core.disturbance import DisturberConfig


@dataclass(frozen=True)
class EvalCondition:
    name: str
    damping: float | None
    disturber: DisturberConfig | None
    bg_cmap: str = "Greys"
    bg_val: float = 0.2
