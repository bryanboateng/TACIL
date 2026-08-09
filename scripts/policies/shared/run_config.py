from dataclasses import dataclass

from scripts.evaluation.core.condition import DisturbanceLevel, FloorCondition


@dataclass(frozen=True)
class RunConfig[PolicyConfigT, AugmentationT]:
    seed: int

    wandb_project: str

    floor_condition: FloorCondition
    disturbance_level: DisturbanceLevel | None

    expert_dataset_artifact: str

    policy: PolicyConfigT
    augmentation: AugmentationT | None
