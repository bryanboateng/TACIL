from dataclasses import dataclass
from typing import Literal


# Where a chain's actions come from.
#   "gpi"         GPI's rule alone, rolled forward through the dynamics model.
#                 A distillation: it turns a policy that has to search the
#                 demonstrations at every step into training data.
#   "gpi_refined" GPI's action is the search's starting point, and the planner
#                 corrects it using the dynamics model. GPI is blind to the
#                 block, so this is where block physics can be added on top of
#                 an action that already moves like the demonstrations.
#   "search"      The planner searches from scratch. Kept for comparison; its
#                 candidate walks carry no prior, and the 2026-08-27d sweeps
#                 showed that costs more than the freedom is worth.
ActionSource = Literal["gpi", "gpi_refined", "search"]


@dataclass(frozen=True)
class RecoveryDataConfig:
    action_source: ActionSource = "gpi_refined"

    # GPI's two weights, as in its policy: how far to follow the recorded
    # action, and how hard to pull toward the recorded position.
    gpi_progression_weight: float = 1.0
    gpi_attraction_weight: float = 1.0

    # Std of the block displacement that is applied to each anchor, as a
    # fraction of each block dimension's range in the expert data. Only the
    # block moves: the evaluation disturbances hit the block, and a displaced
    # agent needs no dynamics model to find its way back.
    disturbance_range_fraction: float = 0.15

    # How far the search may correct GPI's action, as a fraction of each
    # action dimension's range in the expert data. It is a correction radius
    # around an action that already moves like the demonstrations, so it is
    # deliberately small: the planner is here to fix the block, not to invent
    # a different style of motion.
    plan_step_range_fraction: float = 0.02

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
