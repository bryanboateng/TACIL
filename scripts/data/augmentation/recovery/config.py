from dataclasses import dataclass


@dataclass(frozen=True)
class RecoveryDataConfig:
    # Std of the block displacement that is applied to each anchor, as a
    # fraction of each block dimension's range in the expert data. Only the
    # block moves: the evaluation disturbances hit the block, and a displaced
    # agent needs no dynamics model to find its way back.
    disturbance_range_fraction: float = 0.15

    # Std of one step of a candidate action walk, as a fraction of each action
    # dimension's range in the expert data.
    plan_step_range_fraction: float = 0.03

    # How many candidate action sequences the planner scores per chain step.
    candidate_count: int = 128

    # How many steps ahead each candidate sequence looks. It has to be long
    # enough to see a reposition-then-push manoeuvre pay off, and short enough
    # that the model rollout it is judged on still means something.
    plan_horizon: int = 6

    # Hard cap on the recovery chain. A chain that has not arrived at its
    # anchor by then is discarded whole.
    max_chain_length: int = 32

    # The chain counts as arrived when the predicted state is within this many
    # median expert steps of the anchor.
    success_tolerance_step_multiple: float = 1.0

    # Same meaning as in TACIL: ceiling on how far the members' replay
    # endpoints may land from the anchor, in multiples of the median expert
    # step. The planner plans with the ensemble mean; this test replays its
    # chain with every member alone, so a chain that only works where the mean
    # model is wrong does not survive.
    replay_error_limit_step_multiple: float = 6.0
