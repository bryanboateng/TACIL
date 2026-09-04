from dataclasses import dataclass, field
from typing import Annotated, TypeAlias

import tyro

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


@dataclass(frozen=True)
class DataConfig:
    action_noise_range_fraction: float = 0.1

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
        # A limit of zero would reject every step, and the run would stop with
        # no labels at all.
        if self.chain_error_limit_step_multiple <= 0:
            raise ValueError(
                f"chain_error_limit_step_multiple must be positive, but it is "
                f"{self.chain_error_limit_step_multiple}."
            )
