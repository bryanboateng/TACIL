from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    action_noise_range_fraction: float = 0.1

    # Ceiling on the gate signal, in multiples of the median expert step
    # distance. The gate replays the whole candidate chain forward with every
    # ensemble member and measures how far their endpoints spread around the
    # anchor — the uncertainty about what the synthetic episode claims: that
    # these actions lead to the anchor.
    uncertainty_limit_step_multiple: float = 6.0

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
