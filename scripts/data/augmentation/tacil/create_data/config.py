from dataclasses import dataclass

from scripts.data.augmentation.tacil.create_data.chain_error import ChainErrorCriterion


@dataclass(frozen=True)
class DataConfig:
    action_noise_range_fraction: float = 0.1

    # Which rule decides when a chain has grown too far to trust. The rules and
    # what separates them are described in `chain_error`. They are not
    # interchangeable in scale: an accumulated error grows with depth while a
    # per-step one does not, so the limit below has to be searched per rule.
    chain_error_criterion: ChainErrorCriterion = "replay"

    # Ceiling on that error, in multiples of the median expert step distance.
    chain_error_limit_step_multiple: float = 6.0
