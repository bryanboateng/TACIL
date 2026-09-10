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

    def __post_init__(self) -> None:
        for name, value in (
            ("wandb_project", self.wandb_project),
            ("expert_dataset_artifact", self.expert_dataset_artifact),
        ):
            if not value.strip():
                raise ValueError(f"{name} must not be empty.")
