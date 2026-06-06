from dataclasses import dataclass


def _check_unit(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


@dataclass(frozen=True)
class Config:
    model_artifact: str
    dataset_artifact: str

    seed: int = 42
    synthetic_length: int = 5
    expert_length: int = 16
    action_noise_scale: float = 0.1
    label_error_bound_quantile: float = 0.5
    maximum_newton_iterations: int = 30

    def __post_init__(self):
        _check_unit(
            name="label_error_bound_quantile", value=self.label_error_bound_quantile
        )
