from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    wandb_project: str

    model_artifact: str
    dataset_artifact: str

    # Exactly prediction horizon - 1: the deepest policy training window with
    # fully real action labels then starts at the last synthetic state, so the
    # artificial-end rule in the policy dataset (only windows whose labels need no
    # padding) coincides with "every window observes at least one synthetic
    # state". Mini-episodes therefore contribute no window that duplicates the
    # source episode, which is what allows an anchor at every timestep.
    expert_part_length: int = 15

    action_noise_fraction: float = 0.1
    # Fraction of the median expert step distance: caps the estimated
    # deviation of a forward-executed synthetic prefix from its anchor. This
    # is the single conservativeness dial — the error model's conformal scale
    # is a global factor, so a separate calibration knob would be redundant
    # with it.
    error_limit_fraction: float = 0.5

    seed: int = 42
