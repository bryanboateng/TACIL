from dataclasses import dataclass


def _check_unit(*, name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epoch_count: int = 500
    ema_decay: float = 0.9

    def __post_init__(self) -> None:
        _check_unit(name="ema_decay", value=self.ema_decay)


@dataclass(frozen=True)
class ArchitectureConfig:
    hidden_layer_sizes: tuple[int, ...] = (256, 256)
    lipschitz_constant_upper_bound: float | None = None


@dataclass(frozen=True)
class DataSplitConfig:
    seed: int = 0
    validation_fraction: float = 0.2

    def __post_init__(self) -> None:
        _check_unit(name="validation_fraction", value=self.validation_fraction)


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
class Config:
    data: DataConfig
    architecture: ArchitectureConfig
    training: TrainingConfig
    seed: int = 42
