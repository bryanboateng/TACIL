from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class DynamicsConfig:
    hidden_layer_dimensionalities: tuple[int, ...] = (200, 200, 200, 200)
    weight_decays: tuple[float, ...] = (2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4)
    ensemble_size: int = 7
    elite_count: int = 5
    learning_rate: float = 1e-3
    batch_size: int = 256
    holdout_ratio: float = 0.2
    log_variance_loss_coefficient: float = 0.01
    patience_epoch_count: int = 5

    def __post_init__(self) -> None:
        if len(self.weight_decays) != len(self.hidden_layer_dimensionalities) + 1:
            raise ValueError(
                f"weight_decays must hold one value per layer, so "
                f"{len(self.hidden_layer_dimensionalities) + 1} values, but it holds "
                f"{len(self.weight_decays)}."
            )

        if self.elite_count > self.ensemble_size:
            raise ValueError(
                f"elite_count must not exceed ensemble_size {self.ensemble_size}, but "
                f"it is {self.elite_count}."
            )

        if not 0.0 < self.holdout_ratio < 1.0:
            raise ValueError(
                f"holdout_ratio must lie between 0 and 1, but it is "
                f"{self.holdout_ratio}."
            )

        for name, value in (
            ("ensemble_size", self.ensemble_size),
            ("elite_count", self.elite_count),
            ("learning_rate", self.learning_rate),
            ("batch_size", self.batch_size),
            ("patience_epoch_count", self.patience_epoch_count),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, but it is {value}.")


@dataclass(frozen=True)
class RolloutConfig:
    interval_step_count: int = 1_000
    start_state_count: int = 50_000
    length: int = 1
    penalty_coefficient: float = 2.5
    retained_epoch_count: int = 5

    def __post_init__(self) -> None:
        for name, value in (
            ("interval_step_count", self.interval_step_count),
            ("start_state_count", self.start_state_count),
            ("length", self.length),
            ("retained_epoch_count", self.retained_epoch_count),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, but it is {value}.")

        if self.penalty_coefficient < 0:
            raise ValueError(
                f"penalty_coefficient must not be negative, but it is "
                f"{self.penalty_coefficient}."
            )


@dataclass(frozen=True)
class OptimizationConfig:
    hidden_layer_dimensionalities: tuple[int, ...] = (256, 256)
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 3e-4
    temperature_learning_rate: float = 1e-4
    discount_factor: float = 0.99
    target_critic_update_rate: float = 0.005
    batch_size: int = 256
    real_data_ratio: float = 0.05
    epoch_count: int = 3000
    step_count_per_epoch: int = 1000
    compile_learning_step: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("actor_learning_rate", self.actor_learning_rate),
            ("critic_learning_rate", self.critic_learning_rate),
            ("temperature_learning_rate", self.temperature_learning_rate),
            ("discount_factor", self.discount_factor),
            ("target_critic_update_rate", self.target_critic_update_rate),
            ("batch_size", self.batch_size),
            ("epoch_count", self.epoch_count),
            ("step_count_per_epoch", self.step_count_per_epoch),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, but it is {value}.")

        if not 0.0 <= self.real_data_ratio <= 1.0:
            raise ValueError(
                f"real_data_ratio must lie between 0 and 1, but it is "
                f"{self.real_data_ratio}."
            )


@dataclass(frozen=True)
class ValidationConfig:
    interval_epoch_count: int = 100
    episode_count_per_condition: int = 12
    kept_policy: Literal["last", "best"] = "best"

    def __post_init__(self) -> None:
        for name, value in (
            ("interval_epoch_count", self.interval_epoch_count),
            ("episode_count_per_condition", self.episode_count_per_condition),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, but it is {value}.")


@dataclass(frozen=True)
class Config:
    dynamics: DynamicsConfig
    rollout: RolloutConfig
    optimization: OptimizationConfig
    validation: ValidationConfig
