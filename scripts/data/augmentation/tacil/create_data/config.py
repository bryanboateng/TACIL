from dataclasses import dataclass

from scripts.shared.wandb_config import WandbConfig


@dataclass(frozen=True)
class Config:
    wandb: WandbConfig

    model_artifact: str
    dataset_artifact: str

    # How much of the expert trajectory each synthetic prefix is followed by.
    # A diffusion policy trains on whole windows, so a synthetic state is only
    # usable with enough expert continuation behind it: the value must be its
    # prediction horizon - 1, or the augmented episodes feed duplicate windows
    # into policy training.
    #
    # 0 leaves the continuation out, which is what a policy that trains on
    # single (state, action) pairs wants: behavior cloning would otherwise see
    # the expert data repeated once per anchor and the labels drowned in it.
    expert_part_length: int = 15

    action_noise_fraction: float = 0.1
    # Fraction of the median expert step distance
    # It caps how far a synthetic prefix may drift from its anchor.
    error_limit_fraction: float = 0.5

    disagreement_weight: float = 0.5

    # Must match the setting the dynamics model was trained with.
    include_previous_state: bool = False

    seed: int = 42

    def __post_init__(self) -> None:
        if not 0.0 <= self.disagreement_weight <= 1.0:
            raise ValueError(
                f"disagreement_weight must be in [0, 1], got {self.disagreement_weight}"
            )
