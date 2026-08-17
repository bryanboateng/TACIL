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
