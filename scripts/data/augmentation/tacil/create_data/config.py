from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    model_artifact: str
    dataset_artifact: str

    seed: int = 42
    synthetic_part_length: int = 5
    expert_part_length: int = 16
    action_noise_scale: float = 0.1
    label_error_bound_quantile: float = 0.5
    maximum_newton_iterations: int = 30

    def __post_init__(self):
        if self.synthetic_part_length < 1:
            raise ValueError(
                f"synthetic_part_length must be >= 1, got {self.synthetic_part_length}"
            )
        if self.expert_part_length < 1:
            raise ValueError(
                f"expert_part_length must be >= 1, got {self.expert_part_length}"
            )
        if not 0.0 <= self.label_error_bound_quantile <= 1.0:
            raise ValueError(
                f"label_error_bound_quantile must be in [0, 1], got {self.label_error_bound_quantile}"
            )
