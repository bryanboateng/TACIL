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
    interval_epoch_count: int = 40
    episode_count_per_condition: int = 50
    parallel_environment_count: int = 6

    def __post_init__(self) -> None:
        for name, value in (
            ("interval_epoch_count", self.interval_epoch_count),
            ("episode_count_per_condition", self.episode_count_per_condition),
            ("parallel_environment_count", self.parallel_environment_count),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, but it is {value}.")


@dataclass(frozen=True)
class Config:
    optimization: OptimizationConfig

    validation: ValidationConfig | None
    hidden_layer_dimensionality: int = 256

    observation_noise: float | None = None

    def __post_init__(self) -> None:
        if self.hidden_layer_dimensionality <= 0:
            raise ValueError(
                f"hidden_layer_dimensionality must be positive, but it is "
                f"{self.hidden_layer_dimensionality}."
            )

        if self.observation_noise is not None and self.observation_noise <= 0:
            raise ValueError(
                f"observation_noise must be positive when it is set, but it is "
                f"{self.observation_noise}."
            )

        if (
            self.validation is not None
            and self.optimization.epoch_count % self.validation.interval_epoch_count
            != 0
        ):
            raise ValueError(
                f"An epoch count of {self.optimization.epoch_count} must be "
                f"divisible by the validation interval of "
                f"{self.validation.interval_epoch_count}."
                f"Otherwise the final training epochs would not be validated."
            )
