"""One trial of a policy pipeline: download the data, maybe augment it, train,
evaluate.

The run scripts differ in only two things — which policy they train, and which
augmentations that policy can take. Those are the type parameters; everything
else a trial needs is declared here once.
"""

from dataclasses import dataclass

from scripts.evaluation.core.condition import DisturbanceLevel, FloorCondition
from scripts.shared.wandb_config import WandbConfig


@dataclass(frozen=True)
class RunConfig[PolicyConfigT, AugmentationT = None]:
    wandb: WandbConfig

    floor_condition: FloorCondition
    disturbance_level: DisturbanceLevel | None

    expert_dataset_artifact: str

    # The trial's seed. Every stage takes it as an argument rather than holding
    # a copy, because it belongs to the run and not to any one method.
    seed: int

    # How the policy itself is trained, declared in the policy's own config.
    policy: PolicyConfigT

    # None trains on the expert data alone.
    augmentation: AugmentationT | None = None
