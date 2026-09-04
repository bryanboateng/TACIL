from dataclasses import dataclass


@dataclass(frozen=True)
class OptimizationConfig:
    batch_size: int = 512
    epoch_count: int = 200
    learning_rate: float = 1e-3

    def __post_init__(self) -> None:
        for name, value in (
            ("batch_size", self.batch_size),
            ("epoch_count", self.epoch_count),
            ("learning_rate", self.learning_rate),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, but it is {value}.")


@dataclass(frozen=True)
class ValidationConfig:
    interval_epoch_count: int = 10
    episode_count: int = 50

    def __post_init__(self) -> None:
        for name, value in (
            ("interval_epoch_count", self.interval_epoch_count),
            ("episode_count", self.episode_count),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, but it is {value}.")


@dataclass(frozen=True)
class Config:
    optimization: OptimizationConfig
    validation: ValidationConfig
    hidden_layer_dimensionality: int = 256

    def __post_init__(self) -> None:
        if self.hidden_layer_dimensionality <= 0:
            raise ValueError(
                f"hidden_layer_dimensionality must be positive, but it is "
                f"{self.hidden_layer_dimensionality}."
            )

        # Validation should cover the final training epoch. Otherwise, if the
        # validation interval does not divide the epoch count, the final
        # training epochs are never evaluated.
        if self.optimization.epoch_count % self.validation.interval_epoch_count != 0:
            raise ValueError(
                f"An epoch count of {self.optimization.epoch_count} must be "
                f"divisible by the validation interval of "
                f"{self.validation.interval_epoch_count}; otherwise the final "
                f"training epochs would not be validated."
            )
