from dataclasses import dataclass, field
from typing import Annotated, TypeAlias

import tyro


@dataclass(frozen=True)
class SoftSamplingPenalty:
    lipschitz_constraint: float = 2.0
    penalty_weight: float = 0.001
    sampling_standard_deviation: float = 0.0003
    sample_count: int = 5

    def __post_init__(self) -> None:
        for name, value in (
            ("lipschitz_constraint", self.lipschitz_constraint),
            ("sampling_standard_deviation", self.sampling_standard_deviation),
            ("sample_count", self.sample_count),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, but it is {value}.")
        if self.penalty_weight < 0:
            raise ValueError(
                f"penalty_weight must not be negative, but it is {self.penalty_weight}."
            )


@dataclass(frozen=True)
class SpectralNormalization:
    lipschitz_constraint: float = 2.0

    def __post_init__(self) -> None:
        if self.lipschitz_constraint <= 0:
            raise ValueError(
                f"lipschitz_constraint must be positive, but it is "
                f"{self.lipschitz_constraint}."
            )


@dataclass(frozen=True)
class LearnedSlack:
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError(f"weight must be positive, but it is {self.weight}.")


Regularization: TypeAlias = (
    Annotated[SoftSamplingPenalty, tyro.conf.subcommand(name="soft-sampling")]
    | Annotated[SpectralNormalization, tyro.conf.subcommand(name="spectral-norm")]
    | Annotated[LearnedSlack, tyro.conf.subcommand(name="slack")]
)


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
class DynamicsConfig:
    hidden_layer_sizes: tuple[int, ...] = (512, 512)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 512
    epoch_count: int = 100

    regularization: Regularization | None = field(default_factory=SoftSamplingPenalty)

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
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, but it is {value}.")
        if self.weight_decay < 0:
            raise ValueError(
                f"weight_decay must not be negative, but it is {self.weight_decay}."
            )


@dataclass(frozen=True)
class LabelConfig:
    generation: LabelGeneration
    epsilon: float | None

    def __post_init__(self) -> None:
        if self.epsilon is not None and self.epsilon <= 0:
            raise ValueError(
                f"epsilon must be positive when it is set, but it is {self.epsilon}."
            )


@dataclass(frozen=True)
class CcilConfig:
    dynamics: DynamicsConfig
    labels: LabelConfig
