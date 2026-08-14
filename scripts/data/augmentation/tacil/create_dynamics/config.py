from dataclasses import dataclass


@dataclass(frozen=True)
class DynamicsConfig:
    window_length: int
    lipschitz_constant_upper_bound: float | None
    ensemble_size: int = 20
    hidden_layer_count: int = 2
    hidden_layer_dimensionality: int = 256

    def __post_init__(self) -> None:
        if self.ensemble_size < 1:
            raise ValueError(f"ensemble_size must be >= 1, got {self.ensemble_size}")
        if self.hidden_layer_count < 1:
            raise ValueError(
                f"hidden_layer_count must be >= 1, got {self.hidden_layer_count}"
            )
