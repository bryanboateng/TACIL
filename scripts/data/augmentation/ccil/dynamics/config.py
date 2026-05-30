from dataclasses import dataclass

from scripts.data.augmentation.ccil.dynamics_model import ActivationFunctionName


def _check_unit(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 2_048
    epoch_count: int = 500


@dataclass(frozen=True)
class ArchitectureConfig:
    hidden_layer_sizes: tuple[int, ...] = (256, 256)

    # noinspection SpellCheckingInspection
    activation: ActivationFunctionName = "silu"

    lipschitz_constant_upper_bound: float | None = None


@dataclass(frozen=True)
class LossConfig:
    window_length: int = 1
    forward_inverse_ratio: float = 0.5
    inverse_max_newton_iteration_count: int = 10

    def __post_init__(self) -> None:
        _check_unit(name="forward_inverse_ratio", value=self.forward_inverse_ratio)


@dataclass(frozen=True)
class DataSplitConfig:
    seed: int = 0
    holdout_fraction: float = 0.2

    def __post_init__(self) -> None:
        _check_unit(name="holdout_fraction", value=self.holdout_fraction)


@dataclass(frozen=True)
class BackwardSolverConfig:
    max_newton_iteration_count: int = 30
    convergence_tolerance: float = 1e-4
    action_noise_scale: float = 0.1


@dataclass(frozen=True)
class EvaluationConfig:
    data_split: DataSplitConfig
    backward_solver: BackwardSolverConfig
    rollout_horizons: tuple[int, ...] = (1, 2, 3, 5, 9)


@dataclass(frozen=True)
class ExperimentConfig:
    training: TrainingConfig
    architecture: ArchitectureConfig
    loss: LossConfig
    evaluation: EvaluationConfig
    seed: int = 42
