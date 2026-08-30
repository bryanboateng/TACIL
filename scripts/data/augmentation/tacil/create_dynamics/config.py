from dataclasses import dataclass

from scripts.shared.config_checks import check_positive


@dataclass(frozen=True)
class DynamicsConfig:
    window_length: int
    lipschitz_constant_upper_bound: float | None
    ensemble_size: int = 20
    epoch_count: int = 500
    hidden_layer_count: int = 2
    hidden_layer_dimensionality: int = 256

    def __post_init__(self) -> None:
        check_positive(
            window_length=self.window_length,
            ensemble_size=self.ensemble_size,
            epoch_count=self.epoch_count,
            hidden_layer_count=self.hidden_layer_count,
            hidden_layer_dimensionality=self.hidden_layer_dimensionality,
        )
        # None means the model may bend as it likes. A bound of zero would ask
        # for a constant network, and the per-layer bound is a root of it.
        if self.lipschitz_constant_upper_bound is not None:
            check_positive(
                lipschitz_constant_upper_bound=self.lipschitz_constant_upper_bound
            )
