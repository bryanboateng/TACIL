from dataclasses import dataclass, field
from typing import Literal

LipschitzType = Literal["soft_sampling", "spectral_normalization", "none"]
AugmentationType = Literal["noisy_action", "backward_euler_fast"]


@dataclass(frozen=True)
class DynamicsConfig:
    hidden_layer_sizes: tuple[int, ...] = (512, 512)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 512
    epoch_count: int = 100

    lipschitz_type: LipschitzType = "soft_sampling"
    lipschitz_constraint: float = 2.0
    soft_lipschitz_penalty_weight: float = 0.001
    soft_lipschitz_sampling_eps: float = 0.0003
    soft_lipschitz_sample_count: int = 5

    def __post_init__(self) -> None:
        if len(self.hidden_layer_sizes) == 0:
            raise ValueError(
                "hidden_layer_sizes must name at least one size, but it is empty."
            )
        for index, size in enumerate(self.hidden_layer_sizes):
            if size <= 0:
                raise ValueError(
                    f"hidden_layer_sizes must hold positive sizes, but entry "
                    f"{index} is {size}."
                )
        for name, value in (
            ("learning_rate", self.learning_rate),
            ("batch_size", self.batch_size),
            ("epoch_count", self.epoch_count),
            ("lipschitz_constraint", self.lipschitz_constraint),
            ("soft_lipschitz_sampling_eps", self.soft_lipschitz_sampling_eps),
            ("soft_lipschitz_sample_count", self.soft_lipschitz_sample_count),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, but it is {value}.")
        for name, value in (
            ("weight_decay", self.weight_decay),
            ("soft_lipschitz_penalty_weight", self.soft_lipschitz_penalty_weight),
        ):
            if value < 0:
                raise ValueError(f"{name} must not be negative, but it is {value}.")


@dataclass(frozen=True)
class LabelConfig:
    type: AugmentationType = "noisy_action"
    rejection_quantile: float = 0.5
    # Only noisy_action uses these two.
    label_count_per_transition: int = 20
    action_noise_std: float = 1e-5

    def __post_init__(self) -> None:
        if not 0.0 <= self.rejection_quantile <= 1.0:
            raise ValueError(
                f"rejection_quantile must be between 0 and 1, but it is "
                f"{self.rejection_quantile}."
            )
        if self.label_count_per_transition <= 0:
            raise ValueError(
                f"label_count_per_transition must be positive, but it is "
                f"{self.label_count_per_transition}."
            )
        if self.action_noise_std < 0:
            raise ValueError(
                f"action_noise_std must not be negative, but it is "
                f"{self.action_noise_std}."
            )


@dataclass(frozen=True)
class CcilConfig:
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
