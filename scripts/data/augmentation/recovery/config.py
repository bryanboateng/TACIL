from dataclasses import dataclass


@dataclass(frozen=True)
class RecoveryDataConfig:
    # Std of the block displacement that is applied to each anchor, as a
    # fraction of each block dimension's range in the expert data. Only the
    # block moves: the evaluation disturbances hit the block, and a displaced
    # agent needs no dynamics model to find its way back.
    disturbance_range_fraction: float = 0.15

    # Std of one step of a candidate action walk, as a fraction of each action
    # dimension's range in the expert data. It is the starting spread of the
    # search, which then narrows on its own.
    plan_step_range_fraction: float = 0.03

    # How many candidate action sequences are scored per search iteration.
    candidate_count: int = 128

    # How often the search refits its sampling distribution to the best
    # candidates. One iteration is plain random shooting, which is too weak to
    # find the precise pushes this task needs.
    search_iteration_count: int = 4

    # The share of candidates that count as good enough to shape the next
    # iteration's sampling distribution.
    elite_fraction: float = 0.1

    # How many steps ahead each candidate sequence looks. It has to be long
    # enough to see a reposition-then-push manoeuvre pay off, and short enough
    # that the model rollout it is judged on still means something.
    plan_horizon: int = 6

    # How many expert states a chain may aim for. The goal is the demonstrated
    # state distribution, not one particular state, so each chain gets the
    # expert states nearest to where the disturbance left it, and reaching any
    # of them ends the chain. Demanding the anchor itself asks the planner to
    # place the block to the millimetre, which almost never succeeds.
    target_count: int = 32

    # Hard cap on the recovery chain. A chain that has reached no target by
    # then is discarded whole.
    max_chain_length: int = 32

    # A chain counts as arrived when the predicted state is within this many
    # median expert steps of one of its targets. It is also the filter on the
    # start: a disturbance that already leaves the state this close to the
    # demonstrations teaches nothing and is dropped.
    success_tolerance_step_multiple: float = 1.0

    # Same meaning as in TACIL: ceiling on how far the members' replay
    # endpoints may land from the reached target, in multiples of the median
    # expert step. The planner plans with the ensemble mean; this test replays
    # its chain with every member alone, so a chain that only works where the
    # mean model is wrong does not survive.
    replay_error_limit_step_multiple: float = 6.0
