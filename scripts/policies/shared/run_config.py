from dataclasses import dataclass


@dataclass(frozen=True)
class RunConfig[PolicyConfigT, AugmentationT]:
    seed: int

    wandb_project: str

    expert_dataset_artifact: str

    policy: PolicyConfigT
    augmentation: AugmentationT | None

    evaluate_all_conditions: bool = False
