from dataclasses import dataclass

from scripts.shared.wandb_config import WandbConfig


@dataclass(frozen=True)
class Config:
    wandb: WandbConfig

    model_artifact: str
    dataset_artifact: str

    # Must be the diffusion policy's prediction horizon - 1, or the augmented
    # episodes feed duplicate windows into policy training.
    expert_part_length: int = 15

    action_noise_fraction: float = 0.1
    # Fraction of the median expert step distance
    # It caps how far a synthetic prefix may drift from its anchor.
    error_limit_fraction: float = 0.5

    disagreement_weight: float = 0.5

    seed: int = 42

    def __post_init__(self) -> None:
        if not 0.0 <= self.disagreement_weight <= 1.0:
            raise ValueError(
                f"disagreement_weight must be in [0, 1], got {self.disagreement_weight}"
            )
