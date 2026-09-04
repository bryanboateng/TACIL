from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class HorizonLengthConfig:
    observation: int = 2
    prediction: int = 16
    execution: int = 8

    def __post_init__(self) -> None:
        for name, length in (
            ("observation", self.observation),
            ("prediction", self.prediction),
            ("execution", self.execution),
        ):
            if length < 1:
                raise ValueError(
                    f"The {name} horizon is {length}, but must be at least 1."
                )

        # A prediction is aligned with the observation history, so step 0 belongs to
        # the oldest observation and is never run. The current step has index
        # `observation - 1`, and everything executed after it has to fit inside the
        # prediction horizon.

        #     step         0 1 2 3 4 5 6 7 8 9 A B C D E F
        #     observation  o o                                  2 recorded frames
        #     prediction   a a a a a a a a a a a a a a a a     16 predicted actions
        #     executed       x x x x x x x x                    8 actions to run
        #                    ^ current step

        executed_end = self.observation - 1 + self.execution
        if executed_end > self.prediction:
            raise ValueError(
                f"An observation horizon of {self.observation} and an execution "
                f"horizon of {self.execution} need a prediction horizon of at "
                f"least {executed_end}, but it is {self.prediction}. The policy "
                f"would run out of predicted actions before its execution "
                f"horizon is over."
            )


@dataclass(frozen=True)
class OptimizationConfig:
    batch_size: int = 256
    epoch_count: int = 600
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    learning_rate_warmup_step_count: int = 500
    ema_power: float = 0.75

    def __post_init__(self) -> None:
        for name, value in (
            ("batch_size", self.batch_size),
            ("epoch_count", self.epoch_count),
            ("learning_rate", self.learning_rate),
            ("ema_power", self.ema_power),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, but it is {value}.")
        # No decay and no warm-up are both fair choices.
        for name, value in (
            ("weight_decay", self.weight_decay),
            ("learning_rate_warmup_step_count", self.learning_rate_warmup_step_count),
        ):
            if value < 0:
                raise ValueError(f"{name} must not be negative, but it is {value}.")


@dataclass(frozen=True)
class ValidationConfig:
    interval_epoch_count: int = 50
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
    horizon_lengths: HorizonLengthConfig
    optimization: OptimizationConfig
    validation: ValidationConfig
    hidden_channel_counts: tuple[int, ...] = (256, 512, 1024)
    sampler: Literal["ddim", "ddpm"] = "ddim"

    def __post_init__(self) -> None:
        if len(self.hidden_channel_counts) == 0:
            raise ValueError(
                "hidden_channel_counts must name at least one size, but it is empty."
            )
        for index, count in enumerate(self.hidden_channel_counts):
            if count <= 0:
                raise ValueError(
                    f"hidden_channel_counts must hold positive sizes, but entry "
                    f"{index} is {count}."
                )

        # Every hidden level but the last halves the sequence on the way down
        # and doubles it again on the way up. A prediction horizon must be divisible by that factor.
        # Otherwise, it comes back from the network at another length.
        shrink_factor = 2 ** (len(self.hidden_channel_counts) - 1)
        if self.horizon_lengths.prediction % shrink_factor != 0:
            raise ValueError(
                f"A prediction horizon of {self.horizon_lengths.prediction} does "
                f"not divide by {shrink_factor}, the factor by which "
                f"{len(self.hidden_channel_counts)} hidden levels shrink a "
                f"sequence."
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
