"""What a run of the dynamics stage may change.

Only settings worth varying in an ablation live here. Everything else — the
batch sizes, the optimizer, the epoch budget, the data split — is fixed and
sits next to the code that uses it, because a value nobody turns is not
configuration.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DynamicsConfig:
    # The model predicts this number of steps from its own output. Then the
    # loss compares the predictions with the data.
    window_length: int = 3
    # The maximum steepness of the model. If you give None, the model has no
    # Lipschitz layer.
    lipschitz_constant_upper_bound: float | None = None
    # The number of models in the ensemble. The augmentation reads the
    # disagreement between these models, so this sets how fine that signal is.
    ensemble_size: int = 20
    # Shape of the network. Two scalars rather than a tuple of widths: a scalar
    # has an order, so a sweep can tell bigger from smaller and wandb can rank
    # it against the other parameters.
    hidden_layer_count: int = 2
    hidden_layer_dimensionality: int = 256

    def __post_init__(self) -> None:
        if self.ensemble_size < 1:
            raise ValueError(f"ensemble_size must be >= 1, got {self.ensemble_size}")
        if self.hidden_layer_count < 1:
            raise ValueError(
                f"hidden_layer_count must be >= 1, got {self.hidden_layer_count}"
            )
