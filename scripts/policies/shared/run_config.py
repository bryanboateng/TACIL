from dataclasses import dataclass

from scripts.environments.registry import EnvironmentName


@dataclass(frozen=True)
class RunConfig[PolicyConfigT, AugmentationT]:
    seed: int

    wandb_project: str

    environment: EnvironmentName

    expert_dataset_artifact: str

    policy: PolicyConfigT
    augmentation: AugmentationT | None

    evaluate_all_conditions: bool = False
