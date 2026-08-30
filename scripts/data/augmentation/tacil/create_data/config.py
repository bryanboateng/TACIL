from dataclasses import dataclass

from scripts.data.augmentation.tacil.create_data.chain_error import (
    AccountCriterion,
    ChainErrorAccumulation,
    ChainErrorCost,
    ChainErrorCriterion,
    ReplayCriterion,
)


@dataclass(frozen=True)
class DataConfig:
    action_noise_range_fraction: float = 0.1

    # Which rule decides when a chain has grown too far to trust. There are two
    # kinds, and they are described in `chain_error`: replay the finished chain,
    # or charge every step and keep the total. The two fields below say which,
    # and they go together -- both unset is the replay rule, both set picks one
    # account rule out of the cost x accumulation grid.
    #
    # Two flat fields rather than a nested subcommand, which is what this really
    # is: a subcommand would hide every option beside it from the parser, and
    # the sweep has to pass `chain_error_limit_step_multiple` right here.
    chain_error_cost: ChainErrorCost | None = None
    chain_error_accumulation: ChainErrorAccumulation | None = None

    # Ceiling on that error, in multiples of the median expert step distance.
    # The rules are not interchangeable in scale -- an accumulated error grows
    # with depth while a replayed one does not -- so this is searched per rule.
    chain_error_limit_step_multiple: float = 6.0

    def __post_init__(self) -> None:
        if (self.chain_error_cost is None) != (self.chain_error_accumulation is None):
            raise ValueError(
                f"An account rule needs both a cost and an accumulation, and "
                f"the replay rule needs neither, but the cost is "
                f"{self.chain_error_cost!r} and the accumulation is "
                f"{self.chain_error_accumulation!r}."
            )

    @property
    def chain_error_criterion(self) -> ChainErrorCriterion:
        if self.chain_error_cost is None or self.chain_error_accumulation is None:
            return ReplayCriterion()
        return AccountCriterion(
            cost=self.chain_error_cost, accumulation=self.chain_error_accumulation
        )
