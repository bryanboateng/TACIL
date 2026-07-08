from dataclasses import dataclass


@dataclass(frozen=True)
class DataSplitConfig:
    seed: int = 0
    validation_fraction: float = 0.2

    def __post_init__(self) -> None:
        if not 0.0 <= self.validation_fraction <= 1.0:
            raise ValueError(
                f"validation_fraction must be in [0, 1], got {self.validation_fraction}"
            )


@dataclass(frozen=True)
class DataLoaderConfig:
    batch_size: int = 2_048
    window_length: int = 3


@dataclass(frozen=True)
class DataConfig:
    artifact: str
    split: DataSplitConfig
    loader: DataLoaderConfig


@dataclass(frozen=True)
class ArchitectureConfig:
    hidden_layer_sizes: tuple[int, ...] = (256, 256)
    lipschitz_constant_upper_bound: float | None = None
    ensemble_size: int = 10

    def __post_init__(self) -> None:
        if self.ensemble_size < 1:
            raise ValueError(f"ensemble_size must be >= 1, got {self.ensemble_size}")


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epoch_count: int = 500
    ema_decay: float = 0.9

    def __post_init__(self) -> None:
        if not 0.0 <= self.ema_decay <= 1.0:
            raise ValueError(f"ema_decay must be in [0, 1], got {self.ema_decay}")


@dataclass(frozen=True)
class Config:
    wandb_project: str

    data: DataConfig
    architecture: ArchitectureConfig
    training: TrainingConfig
    seed: int = 42
