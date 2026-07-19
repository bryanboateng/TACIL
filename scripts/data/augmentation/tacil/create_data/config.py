from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    wandb_project: str

    model_artifact: str
    dataset_artifact: str

    # Must be the diffusion policy's prediction horizon - 1, or the augmented
    # episodes feed duplicate windows into policy training.
    expert_part_length: int = 15

    action_noise_fraction: float = 0.1
    # Fraction of the median expert step distance
    # It caps how far a synthetic prefix may drift from its anchor.
    error_limit_fraction: float = 0.5

    seed: int = 42
