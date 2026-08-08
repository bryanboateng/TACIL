from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    action_noise_fraction: float = 0.1
    # Fraction of the median expert step distance
    # It caps how far a synthetic prefix may drift from its anchor.
    error_limit_fraction: float = 0.5

    disagreement_weight: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.disagreement_weight <= 1.0:
            raise ValueError(
                f"disagreement_weight must be in [0, 1], got {self.disagreement_weight}"
            )
