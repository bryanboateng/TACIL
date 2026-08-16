from dataclasses import dataclass
from typing import Literal

# Which rule decides that a backward chain has walked far enough.
#
# "per_step" tests each predecessor on its own: the ensemble uncertainty at
# that one step must stay under the limit. It carries nothing from step to
# step; chains still end on their own because the walk drifts away from the
# data and the ensemble's uncertainty rises with depth.
#
# "chain" tests the whole prefix: every ensemble member replays the chain's
# actions forward from the candidate predecessor, and the RMS distance of the
# members' endpoints from the anchor must stay under the limit. An error made
# early passes through every later transition, exactly as it does when the
# synthetic data is executed — so this is the uncertainty about what the data
# claims: that these actions lead to the anchor. With a single step it reduces
# to the per-step test.
GateRule = Literal["per_step", "chain"]


@dataclass(frozen=True)
class DataConfig:
    action_noise_range_fraction: float = 0.1

    # Read as a per-step ceiling under "per_step" and as a ceiling on the whole
    # chain's endpoint spread under "chain". Both are multiples of the median
    # expert step distance, so neither number transfers to the other rule.
    uncertainty_limit_step_multiple: float = 0.5

    gate_rule: GateRule = "chain"

    # How far back a chain may walk before it is stopped regardless of the gate.
    # This is a compute budget, not a claim about the method: training time grows
    # with the number of synthetic steps, and a chain that the gate never stops
    # would walk the length of a whole demonstration. At 6177 anchors that is 1.27
    # million invented transitions against 6417 recorded ones, and a run of about
    # 52 hours instead of one. Fifty steps is six times the length the best runs
    # so far reached, and about five state spreads of travel, so it leaves room
    # for a run-up far longer than any disturbance can call for.
    #
    # It is a parameter rather than a constant so that a run can fix the chain
    # length instead of tuning the gate until it happens to produce that length.
    # That separates two questions the gate otherwise answers at once: whether
    # data of a given depth helps, and whether the gate finds that depth.
    max_synthetic_part_length: int = 50

    def __post_init__(self) -> None:
        if self.max_synthetic_part_length < 1:
            raise ValueError(
                "max_synthetic_part_length must be >= 1, got "
                f"{self.max_synthetic_part_length}"
            )
