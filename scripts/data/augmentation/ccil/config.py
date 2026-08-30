from dataclasses import dataclass, field
from typing import Literal

from scripts.shared.config_checks import (
    check_not_negative,
    check_positive,
    check_positive_sizes,
    check_share,
)

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

    # Bounds only, no opinions: CCIL is a baseline and stays as its paper has
    # it, so these reject values that could not run rather than values this
    # work would not have chosen.
    def __post_init__(self) -> None:
        check_positive_sizes(hidden_layer_sizes=self.hidden_layer_sizes)
        check_positive(
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            epoch_count=self.epoch_count,
            lipschitz_constraint=self.lipschitz_constraint,
            soft_lipschitz_sampling_eps=self.soft_lipschitz_sampling_eps,
            soft_lipschitz_sample_count=self.soft_lipschitz_sample_count,
        )
        check_not_negative(
            weight_decay=self.weight_decay,
            soft_lipschitz_penalty_weight=self.soft_lipschitz_penalty_weight,
        )


@dataclass(frozen=True)
class LabelConfig:
    type: AugmentationType = "noisy_action"
    # Only noisy_action uses these two.
    label_count_per_transition: int = 20
    action_noise_std: float = 1e-5
    rejection_quantile: float = 0.5

    def __post_init__(self) -> None:
        check_positive(label_count_per_transition=self.label_count_per_transition)
        # Zero noise repeats the recorded action, which is a valid label.
        check_not_negative(action_noise_std=self.action_noise_std)
        # A quantile, so it names a share of the labels to drop.
        check_share(rejection_quantile=self.rejection_quantile)


@dataclass(frozen=True)
class CcilConfig:
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    # Named for what the stage produces, rather than after the method, so the
    # command line does not repeat the word twice in one path.
    labels: LabelConfig = field(default_factory=LabelConfig)
