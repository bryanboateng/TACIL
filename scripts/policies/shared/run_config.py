from dataclasses import dataclass

from scripts.environments.registry import EnvironmentName
from scripts.evaluation.config import EvaluationConfig


@dataclass(frozen=True)
class RunConfig[PolicyConfigT, AugmentationT]:
    seed: int

    environment: EnvironmentName

    wandb_project: str

    expert_dataset_artifact: str

    policy: PolicyConfigT
    augmentation: AugmentationT | None

    evaluation: EvaluationConfig

    def __post_init__(self) -> None:
        for name, value in (
            ("wandb_project", self.wandb_project),
            ("expert_dataset_artifact", self.expert_dataset_artifact),
        ):
            if not value.strip():
                raise ValueError(f"{name} must not be empty.")
