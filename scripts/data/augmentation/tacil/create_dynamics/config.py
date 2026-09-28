from dataclasses import dataclass


@dataclass(frozen=True)
class DynamicsConfig:
    window_length: int
    lipschitz_limit: float | None
    ensemble_size: int = 10
    maximum_epoch_count: int = 500
    patience_epoch_count: int = 50
    hidden_layer_count: int = 2
    hidden_layer_dimensionality: int = 256

    def __post_init__(self) -> None:
        for name, value in (
            ("window_length", self.window_length),
            ("ensemble_size", self.ensemble_size),
            ("maximum_epoch_count", self.maximum_epoch_count),
            ("patience_epoch_count", self.patience_epoch_count),
            ("hidden_layer_count", self.hidden_layer_count),
            ("hidden_layer_dimensionality", self.hidden_layer_dimensionality),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, but it is {value}.")
        if self.lipschitz_limit is not None and self.lipschitz_limit <= 0:
            raise ValueError(
                f"lipschitz_limit must be positive, but it is {self.lipschitz_limit}."
            )
