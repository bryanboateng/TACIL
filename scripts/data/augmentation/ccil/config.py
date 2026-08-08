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


@dataclass(frozen=True)
class LabelConfig:
    type: AugmentationType = "noisy_action"
    # Only noisy_action uses these two.
    label_count_per_transition: int = 20
    action_noise_std: float = 1e-5
    rejection_quantile: float = 0.5


@dataclass(frozen=True)
class Config:
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    # Named for what the stage produces, rather than after the method, so the
    # command line does not repeat the word twice in one path.
    labels: LabelConfig = field(default_factory=LabelConfig)
