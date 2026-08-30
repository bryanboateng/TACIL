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
    # `chain_error`. Replaying the finished chain is what TACIL has always
    # done, so it is the default and needs no argument.
    #
    # A subcommand closes the parser around it: every option above has to be
    # given BEFORE `augmentation.data.chain-error:...` on the command line, or
    # it will not be recognised. `create_sweep` places it last for that reason.
    chain_error: ChainErrorArgument = field(default_factory=ReplayCriterion)
