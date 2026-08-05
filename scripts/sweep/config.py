from dataclasses import dataclass

from scripts.evaluation.core.condition import DisturbanceLevel, FloorCondition
from scripts.shared.wandb_config import WandbConfig


@dataclass(frozen=True)
class Config:
    wandb: WandbConfig

    floor_condition: FloorCondition
    disturbance_level: DisturbanceLevel | None

    expert_dataset_artifact: str

    # ---- Swept ----------------------------------------------

    window_length: int
    action_noise_fraction: float
    error_limit_fraction: float
    disagreement_weight: float

    # Left out by the unconstrained sweeps, so the dynamics model is built
    # without a Lipschitz ceiling at all rather than with a stand-in number.
    lipschitz_constant_upper_bound: float | None = None

    # Both carry a default so the command line takes them as flags: the sweep
    # passes the flag when the value is true and omits it otherwise.
    include_previous_state: bool = False
    agent_relative_inputs: bool = False

    seed: int = 42
