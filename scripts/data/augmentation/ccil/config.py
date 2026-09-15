from dataclasses import dataclass
from typing import Annotated, Literal, TypeAlias

import tyro

LipschitzType: TypeAlias = Literal[
    "soft_sampling", "spectral_normalization", "slack", "none"
]

RejectionRule: TypeAlias = Literal["absolute", "quantile", "none"]


@dataclass(frozen=True)
class NoisyActionLabels:
    label_count_per_transition: int = 20
    action_noise_std: float = 1e-5

    def __post_init__(self) -> None:
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
class BackwardEulerLabels:
    pass


@dataclass(frozen=True)
class ForwardEulerLabels:
    max_iteration_count: int = 50

    def __post_init__(self) -> None:
        if self.max_iteration_count <= 0:
            raise ValueError(
                f"max_iteration_count must be positive, but it is "
                f"{self.max_iteration_count}."
            )


LabelGeneration: TypeAlias = (
    Annotated[NoisyActionLabels, tyro.conf.subcommand(name="noisy-action")]
    | Annotated[BackwardEulerLabels, tyro.conf.subcommand(name="backward-euler")]
    | Annotated[ForwardEulerLabels, tyro.conf.subcommand(name="forward-euler")]
)


@dataclass(frozen=True)
class RejectionConfig:
    rule: RejectionRule = "none"
    limit: float = 0.5

    def __post_init__(self) -> None:
        if self.rule == "quantile" and not 0.0 <= self.limit <= 1.0:
            raise ValueError(
                f"A quantile must lie between 0 and 1, but the limit is {self.limit}."
            )

        if self.rule == "absolute" and self.limit <= 0:
            raise ValueError(
                f"A distance must be positive, but the limit is {self.limit}."
            )


@dataclass(frozen=True)
class DynamicsConfig:
    hidden_layer_sizes: tuple[int, ...] = (512, 512)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 512
    epoch_count: int = 100

    lipschitz_type: LipschitzType = "soft_sampling"
    lipschitz_constraint: float = 2.0

    # Read only by soft sampling.
    soft_sampling_penalty_weight: float = 0.001
    soft_sampling_standard_deviation: float = 0.0003
    soft_sampling_sample_count: int = 5

    # Read only by the learned slack.
    slack_weight: float = 1.0

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
            ("soft_sampling_standard_deviation", self.soft_sampling_standard_deviation),
            ("soft_sampling_sample_count", self.soft_sampling_sample_count),
            ("slack_weight", self.slack_weight),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, but it is {value}.")
        for name, value in (
            ("weight_decay", self.weight_decay),
            ("soft_sampling_penalty_weight", self.soft_sampling_penalty_weight),
        ):
            if value < 0:
                raise ValueError(f"{name} must not be negative, but it is {value}.")


@dataclass(frozen=True)
class LabelConfig:
    generation: LabelGeneration
    rejection: RejectionConfig


@dataclass(frozen=True)
class CcilConfig:
    dynamics: DynamicsConfig
    labels: LabelConfig
