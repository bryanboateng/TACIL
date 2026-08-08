from dataclasses import dataclass


@dataclass(frozen=True)
class DataSplitConfig:
    # The seed used to select the validation episodes.
    seed: int = 0
    # This fraction of the episodes goes to validation.
    validation_fraction: float = 0.2

    def __post_init__(self) -> None:
        if not 0.0 <= self.validation_fraction <= 1.0:
            raise ValueError(
                f"validation_fraction must be in [0, 1], got {self.validation_fraction}"
            )


@dataclass(frozen=True)
class DataLoaderConfig:
    # The number of windows in one gradient step.
    batch_size: int = 2_048
    # The model predicts this number of steps from its own output. Then the
    # loss compares the predictions with the data.
    window_length: int = 3


@dataclass(frozen=True)
class DataConfig:
    split: DataSplitConfig
    loader: DataLoaderConfig


@dataclass(frozen=True)
class ArchitectureConfig:
    hidden_layer_sizes: tuple[int, ...] = (256, 256)
    # The maximum steepness of the model. If you give None, the model has no
    # Lipschitz layer.
    lipschitz_constant_upper_bound: float | None = None
    # The number of models in the ensemble. The augmentation reads the
    # disagreement between these models.
    ensemble_size: int = 20

    # Add the previous state to the input. Then the model can read the
    # velocities.
    include_previous_state: bool = False
    # Measure all positions from the agent. If false, the positions start at
    # the corner of the arena.
    measure_positions_from_agent: bool = False

    def __post_init__(self) -> None:
        if self.ensemble_size < 1:
            raise ValueError(f"ensemble_size must be >= 1, got {self.ensemble_size}")


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    # The number of epochs for each model. The training keeps the epoch with
    # the lowest validation loss.
    epoch_count: int = 500
    # The moving average of the weights keeps this fraction of its old value
    # at each update.
    ema_decay: float = 0.9

    def __post_init__(self) -> None:
        if not 0.0 <= self.ema_decay <= 1.0:
            raise ValueError(f"ema_decay must be in [0, 1], got {self.ema_decay}")


@dataclass(frozen=True)
class Config:
    data: DataConfig
    architecture: ArchitectureConfig
    training: TrainingConfig
    # The seed of the run. Each model in the ensemble gets its own seed from
    # this seed.
    seed: int = 42
