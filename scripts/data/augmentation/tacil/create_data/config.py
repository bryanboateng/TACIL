from dataclasses import dataclass, field
from typing import Annotated, Literal, TypeAlias

import tyro

from scripts.data.augmentation.tacil.create_data.branch_actions import ActionSource
from scripts.data.augmentation.tacil.create_data.chain_error import (
    AccountCriterion,
    ReplayCriterion,
)

# The two kinds of rule, as subcommands. The domain types stay free of tyro;
# only this alias, which exists to be parsed, knows about the command line.
ChainErrorArgument: TypeAlias = (
    Annotated[ReplayCriterion, tyro.conf.subcommand(name="replay")]
    | Annotated[AccountCriterion, tyro.conf.subcommand(name="account")]
)


# How the action for a backward step is settled on. The action is what
# decides where the predecessor lands, so it decides what the chain covers.
#
#     random    draw one noisy action and take it, which is what TACIL has
#               always done
#     certain   draw several and keep the one whose predecessor the ensemble
#               agrees on most, so the chain stays in territory the model
#               knows and can spend its error budget on more steps
#     distant   draw several and keep the one whose predecessor lies furthest
#               from the state it was reached from, so each step buys as much
#               new ground as it can
#
# The two named choices are opposite bets on the same budget: spend it on more
# steps, or on bigger ones.
ActionChoice: TypeAlias = Literal["random", "certain", "distant"]


@dataclass(frozen=True)
class DataConfig:
    action_noise_range_fraction: float = 0.1

    # How many actions are drawn per backward step before one is kept. At one
    # there is nothing to choose between and `action_choice` does not matter,
    # which is the behaviour TACIL has always had. Each candidate costs its own
    # inverse dynamics solve, so this multiplies the rollout's cost.
    action_candidate_count: int = 1

    action_choice: ActionChoice = "random"

    # How many chains are grown from each anchor. Each one follows its own
    # starting action, so at more than one the anchor is left in several
    # directions instead of one. The synthetic set grows with this, roughly in
    # proportion, which is part of what it does rather than a side effect.
    branch_count: int = 1

    # Where each branch's starting action comes from; see `branch_actions`. At
    # one branch the two sources agree, because the nearest recorded state to
    # an anchor is the anchor itself.
    action_source: ActionSource = "anchor_noise"

    # Ceiling on the chain error, in multiples of the median expert step
    # distance. The rules are not interchangeable in scale -- an accumulated
    # error grows with depth while a replayed one does not -- so this is
    # searched per rule.
    chain_error_limit_step_multiple: float = 6.0

    # Which rule decides when a chain has grown too far to trust; see
    # `chain_error`. The default is the method's rule: charge every step what
    # the ensemble members disagree by, and add those costs up over the chain.
    # Among the account rules no difference is measurable, so the choice is a
    # design one -- it makes the ensemble the stopping signal rather than only
    # an averaging device.
    #
    # A subcommand closes the parser around it: every plain option has to be
    # given BEFORE the first `...:...` token on the command line, or it will
    # not be recognised, and several subcommands have to be given in the order
    # their fields are declared. `create_sweep` places them last for that
    # reason.
    chain_error: ChainErrorArgument = field(
        default_factory=lambda: AccountCriterion(
            cost="uncertainty", accumulation="summed"
        )
    )

    def __post_init__(self) -> None:
        # Zero noise is a fair choice: it makes every chain follow the anchor
        # action exactly.
        if self.action_noise_range_fraction < 0:
            raise ValueError(
                f"action_noise_range_fraction must not be negative, but it is "
                f"{self.action_noise_range_fraction}."
            )
        if self.branch_count < 1:
            raise ValueError(
                f"branch_count must be at least one, but it is {self.branch_count}."
            )
        if self.action_candidate_count < 1:
            raise ValueError(
                f"action_candidate_count must be at least one, but it is "
                f"{self.action_candidate_count}."
            )
        # A limit of zero would reject every step, and the run would stop with
        # no labels at all.
        if self.chain_error_limit_step_multiple <= 0:
            raise ValueError(
                f"chain_error_limit_step_multiple must be positive, but it is "
                f"{self.chain_error_limit_step_multiple}."
            )
